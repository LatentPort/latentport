"""Live LatentPort feature demos: cached, branch, resume, extend."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import textwrap
import threading
from time import perf_counter

from ttft_demo import Screen, quiet_native, timer_bar

STORIES = {
    "cached": ("KEEP THE CONVERSATION MOVING", "4B reads once -> 27B keeps the conversation", "Ask a follow-up without replaying the document."),
    "branch": ("ONE CONTEXT. TWO SPECIALISTS.", "4B reads once -> translate once -> independent branches", "Give an incident responder and a release planner the same context."),
    "resume": ("YOUR CONTEXT CAN OUTLIVE YOUR SESSION", "Capture -> save to disk -> restore -> ask", "Restore working context and ask a new question."),
    "extend": ("KEEP YOUR MODEL UP TO DATE", "Capture -> append new evidence -> translate -> investigate", "Add a fresh incident update without rereading the original context."),
}


def context(codec, count):
    head = ("ATLAS RELEASE BRIEF. Project codename ORCHID. Release owner Maya. "
            "Deployment Friday at 18:00 UTC. Rollback owner Leon. "
            "Release gate: checkout p95 must remain below 200 milliseconds. "
            "Current incident: checkout p95 is 410 milliseconds. "
            "Suspected cause: expired hot cache keys trigger simultaneous database queries. "
            "Proposed fix: coalesce requests for each hot key. "
            "Security review owner Nia. Customer update owner Theo.\n")
    rows = [head]
    for i in range(count // 10 + 1):
        rows.append(f"Observation {i+1}: checkout latency rises when hot keys expire; "
                    "database query bursts coincide with cache misses. "
                    "The release team is waiting for the latency gate to pass.\n")
    ids = codec.encode(''.join(rows))[:count]
    return codec.decode(ids, special=True)


class Demo:
    def __init__(self, port, screen, output, args):
        self.port, self.screen, self.output, self.args = port, screen, output, args
        self.report = {"success": False, "steps": [], "source": asdict(port.source.identity),
                       "target": asdict(port.target.identity),
                       "method": "Resident GGUF runtime; sequential execution; real committed token callbacks; MTP disabled"}
        self.story = ""
        self.caption = ""
        self.path = ""

    def save(self):
        (self.output/'results.json').write_text(json.dumps(self.report, indent=2)+'\n', encoding='utf-8')

    def trace(self):
        engine = self.port.source.backend._native()
        return {side: engine.trace(side)['prefill_tokens'] for side in ('source','target')}

    def frame(self, task, elapsed, ids, first, text, *, finished=False):
        cols, rows = self.screen.size()
        width = max(20, min(100, cols-6))
        lines = ['LATENTPORT 4+27B / ' + self.story, '='*76, self.path, '']
        lines += textwrap.wrap('TASK: ' + task, width=width)[:3]
        lines += ['', '  ' + timer_bar(elapsed, max(10, math.ceil(elapsed)), max(10,min(46,cols-24)))]
        lines += [f'  FIRST TOKEN: {first:.3f}s | {len(ids)} tokens generated' if first is not None
                  else '  Working...' if not finished else '  Complete.']
        lines += ['', '  LIVE MODEL OUTPUT' if ids else '  ' + self.caption]
        safe = ''.join(c for c in text if c in '\n\t' or ord(c)>=32 and ord(c)!=127)
        wrapped = []
        for paragraph in safe.expandtabs(4).splitlines():
            wrapped.extend(textwrap.wrap(paragraph, width=width) or [''])
        available = max(1, min(12, rows-len(lines)-4))
        lines += ['  '+line for line in wrapped[-available:]] if wrapped else ['  '+('Step complete.' if finished else 'Processing...')]
        return lines

    def job(self, task, operation, *, generating=False):
        before = self.trace()
        data = {'ids': [], 'first': None, 'error': None}
        done = threading.Event()
        started = perf_counter()
        def token(token_id):
            stamp = perf_counter()
            if data['first'] is None: data['first'] = stamp-started
            data['ids'].append(token_id)
        def work():
            try:
                data['value'] = operation(token)
            except BaseException as exc:
                data['error'] = exc
            finally:
                data['elapsed'] = perf_counter()-started
                done.set()
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        if not self.screen.live: self.screen.say(task)
        while not done.wait(.08):
            ids = list(data['ids'])
            text = self.port.target.codec.decode(ids, special=True) if ids else ''
            self.screen.draw(self.frame(task, perf_counter()-started, ids, data['first'], text))
        worker.join()
        if data['error']: raise data['error']
        text = self.port.target.codec.decode(data['ids'], special=True) if data['ids'] else ''
        after = self.trace()
        entry = {'demo': self.args.demo, 'task': task, 'elapsed_s': data['elapsed'],
                 'observed_ttft_s': data['first'], 'token_ids': data['ids'], 'text': text,
                 'prefill_tokens': {k: after[k]-before[k] for k in before}}
        value = data['value']
        if generating:
            entry.update(finished=value.finished, stats=asdict(value.stats))
        self.report['steps'].append(entry)
        self.save()
        self.screen.draw(self.frame(task, data['elapsed'], data['ids'], data['first'], text, finished=True),final=True)
        if generating and not self.args.yes:
            self.screen.say('Press Enter to continue.')
            if not sys.stdin.readline(): raise RuntimeError('No input. Use --yes for unattended runs.')
        return value, entry

    def answer(self, session, question, *, complete=False):
        def operation(callback):
            answer = session.ask(question, max_new_tokens=min(192,self.args.answer_tokens), mtp=False,on_token=callback)
            used = len(answer.token_ids)
            while complete and not answer.finished and used < self.args.answer_tokens:
                answer = session.generate(max_new_tokens=min(192,self.args.answer_tokens-used), mtp=False,on_token=callback)
                used += len(answer.token_ids)
            if complete and not answer.finished:
                raise RuntimeError('The first answer reached its token budget. Increase --answer-tokens to complete the turn.')
            return answer
        return self.job(question.rsplit("\n",1)[-1] if len(question)>500 else question,operation,generating=True)

    def checked(self, condition, message):
        if not condition: raise RuntimeError('Verification failed: '+message)
        self.report.setdefault('verified',[]).append(message)
        self.save()

    def begin(self, name):
        self.args.demo = name
        self.story,self.path,self.caption = STORIES[name]
        self.screen.draw(['LATENTPORT 4+27B', '='*76,'',self.story,'',self.path,'',
                          *textwrap.wrap(self.caption,76),'',
                          '  CONTEXT: Atlas release brief and checkout incident observations.',
                          '  Follow the task, watch the tokens, see the verified state operations.',
                          '', '  READY TO BEGIN', '  Press Enter to begin.'],final=True)
        if not self.args.yes and not sys.stdin.readline():
            raise RuntimeError('No input. Use --yes for unattended runs.')

    def capture(self, text):
        state, entry = self.job('Read the Atlas brief with 4B and capture its working state.',
                               lambda _:self.port.source.capture([('user',text)]))
        self.checked(entry['prefill_tokens']['target']==0,'Context captured without target prefill')
        return state

    def translate(self, state):
        result, entry = self.job('Transfer the prepared context from 4B to 27B.',
                                 lambda _:self.port.bridge.translate(state))
        self.checked(all(v==0 for v in entry['prefill_tokens'].values()),'Bridge installed state without prompt replay')
        return result

    def run(self, name, text):
        self.begin(name)
        if name == 'cached':
            chat = self.port.composite.session()
            first,e1 = self.answer(chat,text+'\nReply with only the project codename.',complete=True)
            self.checked(e1['prefill_tokens']['target']==0,'First composite prompt processed only by 4B')
            del first
            second,e2 = self.answer(chat,'Who owns rollback? Reply with just the name.')
            self.checked(e2['prefill_tokens']['source']==0,'Follow-up kept the reader idle')
            self.checked(0<e2['prefill_tokens']['target']<128,'Follow-up processed only a short new target suffix')
            self.report['cached_ttft']={'initial_s':e1['observed_ttft_s'],'followup_s':e2['observed_ttft_s']}
            highlights=[f"Initial TTFT: {e1['observed_ttft_s']:.3f}s",f"Follow-up TTFT: {e2['observed_ttft_s']:.3f}s",
                        f"Follow-up prompt tokens: {e2['prefill_tokens']['target']} | 4B: 0",'Conversation state retained. No history replay.']
        elif name == 'branch':
            source = self.capture(text)
            shared = self.translate(source)
            del source
            branches,_ = self.job('Fork the prepared context into two independent specialists.',
                                  lambda _:self.port.target.session(shared).fork(2))
            self.checked(branches[0].state.state_id!=branches[1].state.state_id,'Branches have independent state identities')
            untouched = branches[1].state.state_id
            a,e1 = self.answer(branches[0],'You are the incident responder. What is the likely cause and one fix? Be concise.')
            self.checked(branches[1].state.state_id==untouched,'First specialist did not advance the second branch')
            del a
            b,e2 = self.answer(branches[1],'You are the release planner. Name the release gate and rollback owner. Be concise.')
            for e in (e1,e2):
                self.checked(e['prefill_tokens']['source']==0 and 0<e['prefill_tokens']['target']<128,'Specialist processed only its new question')
            highlights=['ONE source capture. ONE translation. TWO independent branches.',
                        f"Incident responder TTFT: {e1['observed_ttft_s']:.3f}s",
                        f"Release planner TTFT: {e2['observed_ttft_s']:.3f}s",'Both specialists reused the prepared context.']
        elif name == 'resume':
            import latentport as lp
            path = self.args.state
            if path is None:
                state = self.capture(text)
                path = self.output/'atlas.lpstate'
                _,saved = self.job('Save the working context to disk.',lambda _:state.save(path))
                del state
                gc.collect()
                self.report['saved_state']=str(path)
            restored,loaded = self.job('Load the saved state without replaying the Atlas brief.',lambda _:lp.State.load(path))
            self.checked(all(v==0 for v in loaded['prefill_tokens'].values()),'Disk restore performed zero prompt prefill')
            shared = self.translate(restored)
            del restored
            answer,e = self.answer(self.port.target.session(shared),'What is the project codename and who owns rollback? Be concise.')
            self.checked(e['prefill_tokens']['source']==0 and 0<e['prefill_tokens']['target']<128,'Restored context answered a new question without historical replay')
            highlights=[f"State loaded in {loaded['elapsed_s']:.3f}s",'Historical prompt tokens replayed: 0',
                        f"New-question TTFT: {e['observed_ttft_s']:.3f}s",'Saved context is ready for another session.']
        elif name == 'extend':
            original = self.capture(text)
            original_clock = original.clock
            update='\nLATEST INCIDENT UPDATE: Request coalescing is deployed. Checkout p95 fell from 410 ms to 120 ms. Maya approved the release gate.\n'
            messages = original.messages + (('user',update),)
            rendered = self.port.source.codec.render(messages,generation_prompt=False)
            prior = original.rendered_token_ids
            if rendered.token_ids[:len(prior)] != prior:
                raise RuntimeError('Update template would rewrite history; refusing replay')
            suffix = rendered.token_ids[len(prior):]
            added,entry = self.job('Append fresh telemetry: checkout p95 is now 120 ms; release gate approved.',
                                  lambda _:self.port.source.append(original,suffix,messages=messages))
            expected=len(suffix)
            self.checked(entry['prefill_tokens']['source']==expected and entry['prefill_tokens']['target']==0,
                         'Reader processed exactly the newly appended tokens')
            self.checked(original.clock==original_clock and added.clock==original_clock+expected,'Append preserved the original snapshot')
            del original
            shared=self.translate(added)
            answer,e=self.answer(self.port.target.session(shared),'What changed in the latest update? Does checkout now pass the release gate? Be concise.')
            highlights=[f"Original context: {original_clock:,} tokens",f"New evidence processed: {expected} tokens",'Original snapshot preserved.',
                        'Updated working state transferred to 27B.']
        self.report.setdefault('highlights',{})[name]=highlights
        self.save()
        self.screen.draw(['LATENTPORT 4+27B / '+self.story,'='*76,'',
                          *['  '+line for line in highlights],'','  READ. TRANSFER. REUSE.',''],final=True)
        if not self.args.yes:
            self.screen.say('Press Enter to finish this demo.')
            sys.stdin.readline()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('demo',choices=[*STORIES,'all'])
    parser.add_argument('--context-tokens',type=int,default=4096)
    parser.add_argument('--answer-tokens',type=int,default=2048,help='Maximum tokens to finish the first cached conversation turn')
    parser.add_argument('--state',type=Path,help='Existing source state to restore in the resume demo')
    parser.add_argument('--yes',action='store_true')
    args=parser.parse_args()
    if not 256<=args.context_tokens<=8192:parser.error('context-tokens must be 256..8192')
    if not 192<=args.answer_tokens<=4096:parser.error('answer-tokens must be 192..4096')
    if args.state and args.demo!='resume':parser.error('--state is for resume only')
    import latentport as lp
    from latentport.hub import bulk_home
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=bulk_home()/'demos'/f'state-{stamp}-{os.getpid()}'
    output.mkdir(parents=True,exist_ok=False)
    port=None
    with quiet_native(output/'native.log') as console:
        screen=Screen(console)
        demo=None
        try:
            screen.say('LATENTPORT / PREPARING YOUR LIVE STATE DEMO')
            screen.say(f'Logs: {output / "native.log"}')
            port=lp.load(backend='gguf');port.warm()
            demo=Demo(port,screen,output,args)
            text=context(port.source.codec,args.context_tokens)
            (output/'context.txt').write_text(text,encoding='utf-8')
            names=list(STORIES) if args.demo=='all' else [args.demo]
            for name in names:
                demo.run(name,text)
                gc.collect()
            demo.report['success']=True;demo.save()
            screen.say(f'Evidence: {output / "results.json"}')
        except BaseException as exc:
            if demo:
                demo.report['error']=f'{type(exc).__name__}: {exc}';demo.save()
            screen.say(f'DEMO STOPPED: {exc}')
            raise
        finally:
            if port:port.close()


if __name__=='__main__':main()
