"""Authenticated, process-local sessions backed by vLLM streaming inputs."""

import asyncio
import hashlib
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, StrictInt
from vllm import envs
from vllm.engine.protocol import StreamingInput
from vllm.sampling_params import RequestOutputKind, SamplingParams

from .settings import CONFIG_KEY, SESSION_PLUGIN, SESSION_TTL_DEFAULT

logger = logging.getLogger("vllm.latentport")
TURN_TIMEOUT_SECONDS = 900


class TurnInput(BaseModel):
    turn: Annotated[int, Field(strict=True, ge=1)]
    token_ids: Annotated[list[StrictInt], Field(min_length=1, max_length=262144)]
    max_tokens: Annotated[int, Field(strict=True, ge=1, le=512)] = 64


@dataclass
class PendingTurn:
    number: int
    input_count: int
    future: asyncio.Future


class Session:
    def __init__(self, store, owner):
        self.store = store
        self.owner = owner
        self.id = secrets.token_urlsafe(24)
        self.next_turn = 1
        self.last_turn = None
        self.logical_tokens = 0
        self.pending = None
        self.chunks = asyncio.Queue(maxsize=1)
        self.loop = asyncio.get_running_loop()
        self.timer = self.loop.call_later(store.ttl, self.expire)
        self.closing = False
        self.task = self.loop.create_task(self.run())
        self.task.add_done_callback(self.finished)

    def describe(self):
        return {
            "id": self.id,
            "next_turn": self.next_turn,
            "logical_tokens": self.logical_tokens,
            "busy": self.pending is not None,
            "idle_ttl_seconds": self.store.ttl,
            "last_turn": self.last_turn,
        }

    def expire(self):
        self.closing = True
        self.task.cancel()

    def finished(self, _):
        self.timer.cancel()
        if self.pending is not None and not self.pending.future.done():
            self.pending.future.set_result(None)
        self.store.sessions.pop(self.id, None)

    async def close(self):
        if not self.closing:
            self.expire()
        await asyncio.gather(self.task, return_exceptions=True)

    async def inputs(self):
        while True:
            yield await self.chunks.get()

    async def run(self):
        output_ids = []
        output_text = []
        try:
            async for output in self.store.engine.generate(
                self.inputs(),
                SamplingParams(output_kind=RequestOutputKind.DELTA, temperature=0),
                request_id=f"lp-session-{self.id}",
            ):
                completion = output.outputs[0]
                output_ids.extend(completion.token_ids)
                output_text.append(completion.text)
                if completion.finish_reason is None:
                    continue
                pending = self.pending
                if completion.finish_reason not in ("stop", "length") or not output_ids:
                    raise RuntimeError("vLLM interrupted the session turn")
                self.logical_tokens += pending.input_count + len(output_ids)
                self.last_turn = {
                    "turn": pending.number,
                    "text": "".join(output_text),
                    "token_ids": output_ids,
                    "finish_reason": completion.finish_reason,
                    "logical_tokens": self.logical_tokens,
                }
                self.next_turn += 1
                self.pending = None
                self.timer.cancel()
                self.timer = self.loop.call_later(self.store.ttl, self.expire)
                pending.future.set_result(self.last_turn)
                output_ids = []
                output_text = []
        except Exception:
            logger.exception("LPCompact resident session failed")

    async def append(self, turn):
        if self.pending is not None:
            raise HTTPException(409, "Another turn is in progress")
        if turn.turn != self.next_turn:
            raise HTTPException(409, {"next_turn": self.next_turn})
        if self.logical_tokens + len(turn.token_ids) + turn.max_tokens > self.store.max_len:
            raise HTTPException(422, "Turn exceeds the model context limit")
        if any(token < 0 or token >= self.store.vocab_size for token in turn.token_ids):
            raise HTTPException(422, "Token ID outside the model vocabulary")

        # vLLM drops the previous turn's final sampled token because it has
        # never run through the model. Materialize it ahead of the new suffix.
        edge = self.last_turn["token_ids"][-1:] if self.last_turn else []
        params = SamplingParams(
            max_tokens=turn.max_tokens,
            temperature=0,
            output_kind=RequestOutputKind.DELTA,
        )
        future = self.loop.create_future()
        self.pending = PendingTurn(turn.turn, len(turn.token_ids), future)
        self.timer.cancel()
        self.timer = self.loop.call_later(TURN_TIMEOUT_SECONDS, self.expire)
        self.chunks.put_nowait(
            StreamingInput(
                prompt={"prompt_token_ids": [*edge, *turn.token_ids]},
                sampling_params=params,
            )
        )
        result = await asyncio.shield(future)
        if result is None:
            raise HTTPException(503, "Session ended before the turn completed")
        return result


class SessionStore:
    def __init__(self, engine, options):
        self.engine = engine
        self.ttl = options.get("session_ttl", SESSION_TTL_DEFAULT)
        self.limit = options["max_sessions"]
        self.max_len = engine.model_config.max_model_len
        self.vocab_size = engine.model_config.get_vocab_size()
        self.sessions = {}

    @staticmethod
    def owner(request):
        # Native vLLM middleware authenticates the /v1 prefix. Bind sessions
        # to the particular verified key when several server keys are used.
        _, _, token = request.headers["authorization"].partition(" ")
        return hashlib.sha256(token.encode()).digest()

    def create(self, request):
        if len(self.sessions) >= self.limit:
            raise HTTPException(429, "Resident session capacity is full")
        session = Session(self, self.owner(request))
        self.sessions[session.id] = session
        return session

    def lookup(self, request, session_id):
        session = self.sessions.get(session_id)
        if (
            session is None
            or session.closing
            or not secrets.compare_digest(session.owner, self.owner(request))
        ):
            raise HTTPException(404, "Session not found")
        return session

    async def close(self):
        await asyncio.gather(*(session.close() for session in self.sessions.values()))


class SessionEndpoints:
    name = SESSION_PLUGIN
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI):
        router = APIRouter(prefix="/v1/latentport/sessions")

        @router.post("", status_code=201)
        async def create(request: Request):
            return request.app.state.latentport_sessions.create(request).describe()

        @router.get("/{session_id}")
        async def status(request: Request, session_id: str):
            return request.app.state.latentport_sessions.lookup(request, session_id).describe()

        @router.post("/{session_id}/turns")
        async def append(request: Request, session_id: str, turn: TurnInput):
            session = request.app.state.latentport_sessions.lookup(request, session_id)
            return await session.append(turn)

        @router.delete("/{session_id}", status_code=204)
        async def close(request: Request, session_id: str):
            await request.app.state.latentport_sessions.lookup(request, session_id).close()
            return Response(status_code=204)

        app.include_router(router)
        lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def cleanup(serving_app):
            async with lifespan(serving_app) as state:
                try:
                    yield state
                finally:
                    await serving_app.state.latentport_sessions.close()

        app.router.lifespan_context = cleanup

    async def init_state(self, engine_client, state, args):
        options = engine_client.vllm_config.additional_config.get(CONFIG_KEY, {})
        if not options.get("max_sessions"):
            raise ValueError("Enable resident sessions with --lpcompact-max-sessions")
        if not any(args.api_key or [envs.VLLM_API_KEY]):
            raise ValueError("Resident sessions require --api-key or VLLM_API_KEY")
        if engine_client.vllm_config.parallel_config._api_process_count != 1:
            raise ValueError("Resident sessions require one API server process")
        state.latentport_sessions = SessionStore(engine_client, options)
