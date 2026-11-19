import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from fastapi import HTTPException

from latentport_vllm.sessions import Session, TurnInput


def test_session_reuses_last_edge_and_cancels_resident_request():
    async def check():
        submitted = []
        cancelled = asyncio.Event()

        class Engine:
            async def generate(self, chunks, params, request_id):
                try:
                    async for chunk in chunks:
                        submitted.append(chunk.prompt["prompt_token_ids"])
                        for token, text, finish in ((50, "A", None), (51, "B", "length")):
                            yield SimpleNamespace(outputs=[SimpleNamespace(
                                token_ids=[token], text=text, finish_reason=finish
                            )])
                finally:
                    cancelled.set()

        store = SimpleNamespace(
            engine=Engine(), ttl=10, max_len=12, vocab_size=100, sessions={}
        )
        session = Session(store, b"owner")
        store.sessions[session.id] = session
        try:
            result = await session.append(TurnInput(turn=1, token_ids=[2, 3], max_tokens=2))
            assert result["logical_tokens"] == 4 and result["text"] == "AB"
            with pytest.raises(HTTPException) as stale:
                await session.append(TurnInput(turn=1, token_ids=[4], max_tokens=2))
            assert stale.value.status_code == 409
            result = await session.append(TurnInput(turn=2, token_ids=[4], max_tokens=2))
            assert result["logical_tokens"] == 7
            assert submitted == [[2, 3], [51, 4]]
            with pytest.raises(HTTPException) as overflow:
                await session.append(TurnInput(turn=3, token_ids=[1, 2], max_tokens=4))
            assert overflow.value.status_code == 422
        finally:
            await session.close()
        assert cancelled.is_set()
        assert not store.sessions

    asyncio.run(check())


def test_idle_session_expiry_releases_the_quota():
    async def check():
        async def generate(chunks, params, request_id):
            async for chunk in chunks:
                yield chunk

        store = SimpleNamespace(
            engine=SimpleNamespace(generate=generate), ttl=0.001, sessions={}
        )
        session = Session(store, b"owner")
        store.sessions[session.id] = session
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(session.task, timeout=1)
        assert not store.sessions

    asyncio.run(check())
