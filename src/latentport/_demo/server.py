"""Local flagship demo. Run with the installed LatentPort GGUF environment."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import webbrowser

from .logging import quiet_native
from .fixture import build_fixture,run_tests,extract_json,verify

TASK='Find why the checkout charges $1.19 for two $59.95 headphones. Identify the exact faulty line and explain the fix in at most three sentences. Do not write the patch yet.'
PATCH_TASK='You are the repair specialist. Return one JSON object with keys file, old, new. file must be shop/checkout.py. old and new must each be a single assignment line with no indentation. Repair amount_minor using the existing imported to_minor_units helper and the current total and currency variables. Do not change any other code. Return JSON only.'
TEST_TASK='You are the regression specialist. Return JSON with a cases array of exactly three independent checkout test cases. Each case has price (decimal string), quantity (integer), currency, expected_minor (integer). Include USD price 59.95 quantity 2, JPY price 1200 quantity 2, and one KWD case. Compute expected minor-unit amounts using the money contract in the repository. Return JSON only.'

class App:
    def __init__(self,output,args):
        self.output,self.args=output,args
        self.lock=threading.RLock();self.port=None;self.active=False;self.continue_event=threading.Event()
        self.data={'status':'loading','phase':'loading','message':'Loading the models onto your GPU',
                   'title':'TELEPATHY FOR LLMS','task':TASK,'tokens':0,'files':[],
                   'stages':{'read':'waiting','transfer':'waiting','generate':'waiting','fork':'waiting','verify':'waiting'},
                   'runs':{k:{'text':'','tokens':0,'ttft':None,'started':None,'status':'waiting'} for k in ('native','composite','repair','regression')},
                   'checks':[],'events':[],'context_tokens':None,'speedup':None,'result':None}
        self.report={'success':False,'method':'Sequential native/composite runs; same fresh prompt; both paths warmed; resident models; greedy decoding; MTP off; external request-to-first-committed-token timing',
                     'scope':'TTFT excludes downloads, model loading, warmups; first token can be reasoning/control; no full-response speed claim',
                     'fixture':'Local generated checkout demonstration repository; no real payment services',
                     'runs':{},'checks':[]}

    def set(self,**values):
        with self.lock:self.data.update(values)
    def event(self,phase,message):
        with self.lock:
            self.data.update(phase=phase,message=message)
            self.data['events'].append({'phase':phase,'message':message,'time':time.time()})
    def stage(self,key,value):
        with self.lock:self.data['stages'][key]=value
    def snapshot(self):
        with self.lock:return json.loads(json.dumps(self.data))
    def save(self):
        (self.output/'results.json').write_text(json.dumps(self.report,indent=2)+'\n',encoding='utf-8')
    def check(self,condition,label):
        if not condition:raise RuntimeError(label)
        with self.lock:self.data['checks'].append(label)
        self.report['checks'].append(label)
    def trace(self):
        engine=self.port.source.backend._native()
        return {k:engine.trace(k)['prefill_tokens'] for k in ('source','target')}
    def cleanup(self):
        import torch
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()

    def prepare(self):
        try:
            import latentport as lp
            from ..diagnostics import diagnose
            from ..hub import gguf_model_file, gguf_bridge_file, gguf_model_specs
            self.event('checking', 'Checking dependencies and GPU compatibility')
            report = diagnose('gguf', check_native=False)
            if report['errors']:
                raise RuntimeError('; '.join(report['errors']))
            if report['gpu_total_bytes'] < 30 * 1024**3:
                raise RuntimeError('The transfer demo requires a qualified GPU with at least 32 GB VRAM.')
            from ..hub import transfer_download_bytes, check_download_space
            required = transfer_download_bytes(self.args.model_manifest)
            check_download_space(required)
            self.report['download_bytes_missing'] = required
            self.event('installing', 'Checking the native runtime and CUDA libraries')
            from .._native.engine import Library
            try:
                Library(None)
            except lp.CapabilityError:
                if os.environ.get('LATENTPORT_NATIVE_LIB'):
                    raise
                from ..__main__ import install_native
                install_native()
                Library(None)
            specs = gguf_model_specs(self.args.model_manifest)
            paths = {}
            for side in ('source', 'target'):
                self.event('downloading', f"Preparing {side} model ({specs[side]['bytes'] / 1e9:.2f} GB); cached files are reused")
                paths[side] = gguf_model_file(side, manifest_path=self.args.model_manifest)
            self.event('downloading', 'Preparing the bridge (2.58 GB); cached files are reused')
            bridge_path = gguf_bridge_file()
            self.event('loading', 'Loading the models and bridge onto your GPU')
            self.repo=self.output/'original-checkout'
            bundle,files=build_fixture(self.repo)
            self.prompt='Analyze this repository. File contents follow.\n\n'+bundle+'\n\nTASK: '+TASK
            (self.output/'prompt.txt').write_text(self.prompt,encoding='utf-8')
            incident=run_tests(self.repo)
            if incident['passed']:raise RuntimeError('Fixture failed to reproduce the checkout incident')
            self.report['original_tests']=incident
            self.set(files=[{'name':name,'text':text} for name,text in files.items()],original_tests=incident)
            self.port=lp.load(backend='gguf', source_path=paths['source'], target_path=paths['target'],
                              bridge_gguf_path=bridge_path, native_options=lp.NativeOptions(mtp_drafts=0))
            self.report['residency']=self.port.warm()
            count=len(self.port.source.codec.render([('user',self.prompt)],generation_prompt=True).token_ids)
            if count>16000:raise RuntimeError(f'Repository exceeds the demo context budget: {count}')
            self.report.update(context_tokens=count,source=asdict(self.port.source.identity),target=asdict(self.port.target.identity))
            self.set(context_tokens=count)
            self.event('warming','Warming both execution paths. The live run starts fresh.')
            for name in ('native','composite'):
                self.cleanup()
                session=self.port.target.session() if name=='native' else self.port.composite.session()
                answer=session.ask(self.prompt,max_new_tokens=1,mtp=False)
                del answer,session
            self.cleanup()
            self.set(status='ready');self.event('ready','READY TO BEGIN')
            self.save()
            if self.args.headless:self.start()
        except BaseException as exc:self.fail(exc)

    def fail(self,exc):
        self.set(status='error',message=str(exc));self.report['error']=f'{type(exc).__name__}: {exc}'
        self.report['traceback']=traceback.format_exc();self.save();self.active=False

    def stream(self,key,session,*,question=None,prepare=None,complete=False):
        ids=[];first=None;last_update=0
        before=self.trace();started=time.perf_counter()
        with self.lock:self.data['runs'][key].update(started=time.time(),status='running')
        def callback(token):
            nonlocal first,last_update
            stamp=time.perf_counter()
            if first is None:first=stamp-started
            ids.append(token)
            if stamp-last_update>.06 or len(ids)==1:
                text=self.port.target.codec.decode(ids,special=True)
                with self.lock:self.data['runs'][key].update(text=text,tokens=len(ids),ttft=first)
                last_update=stamp
        if prepare is not None:session=prepare()
        if question is None:
            answer=session.generate(max_new_tokens=192,mtp=False,on_token=callback)
        else:
            answer=session.ask(question,max_new_tokens=192,mtp=False,on_token=callback)
        initial_stats=asdict(answer.stats)
        while complete and not answer.finished and len(ids)<self.args.max_tokens:
            answer=session.generate(max_new_tokens=min(192,self.args.max_tokens-len(ids)),mtp=False,on_token=callback)
        text=self.port.target.codec.decode(ids,special=True)
        with self.lock:self.data['runs'][key].update(text=text,tokens=len(ids),ttft=first,status='done')
        if complete and not answer.finished:raise RuntimeError(f'{key} reached its generation budget; increase --max-tokens')
        after=self.trace();delta={k:after[k]-before[k] for k in before}
        self.report['runs'][key]={'ttft_s':first,'elapsed_s':time.perf_counter()-started,'token_ids':ids,'text':text,
                                  'finished':answer.finished,'prefill_tokens':delta,'initial_call_stats':initial_stats}
        self.save()
        return session,text,delta

    def start(self):
        with self.lock:
            if self.data['status']!='ready' or self.active:return False
            self.active=True;self.data['status']='running'
        threading.Thread(target=self.run,daemon=True).start()
        return True

    def checkpoint(self,phase,message):
        if self.args.headless:return
        self.continue_event.clear()
        self.event(phase,message)
        self.set(status='paused')
        self.continue_event.wait()

    def advance(self):
        with self.lock:
            if self.data['status']!='paused':return False
            self.data['status']='running'
            self.continue_event.set()
        return True

    def run(self):
        try:
            self.event('native','First, the native 27B reads the repository.')
            self.cleanup()
            native,text,delta=self.stream('native',self.port.target.session(),question=self.prompt)
            expected=len(self.port.target.codec.render([('user',self.prompt)],generation_prompt=True).token_ids)
            self.check(delta=={'source':0,'target':expected},'Native baseline processed the entire fresh prompt')
            del native
            self.checkpoint('native_complete','Native 27B read the code. Now let the small model do the reading.')
            self.cleanup()
            self.event('read','Now the 4B reads the same repository and task.')
            self.stage('read','active')
            def handoff():
                source=self.port.source.capture([('user',self.prompt)],generation_prompt=True)
                self.stage('read','done');self.stage('transfer','active')
                self.event('transfer','Translating full hybrid state: attention, recurrent memory, convolution and hidden state.')
                t=time.perf_counter();target=self.port.bridge.translate(source)
                self.report['translation_s']=time.perf_counter()-t
                del source
                self.stage('transfer','done');self.stage('generate','active')
                self.event('generate','27B is generating from the inherited state.')
                return self.port.target.session(target)
            shared,text,delta=self.stream('composite',None,prepare=handoff,complete=True)
            self.check(delta['target']==0,'27B prompt tokens during composite request: 0')
            self.check(delta['source']==self.data['context_tokens'],'4B processed the entire fresh composite prompt')
            self.stage('generate','done')
            ratio=self.report['runs']['native']['ttft_s']/self.report['runs']['composite']['ttft_s']
            self.report['ttft_speedup']=ratio;self.set(speedup=ratio)
            self.checkpoint('composite_complete','27B answered from the transferred state. Now split that context into two tasks.')
            self.event('fork','One inherited context. Two independent specialists.')
            self.stage('fork','active')
            repair,regression=shared.fork(2)
            untouched=regression.state.state_id
            self.check(repair.state.state_id!=untouched,'Forked sessions have independent state identities')
            self.stage('fork','done')
            self.event('repair','Repair specialist: propose the smallest correct patch.')
            repair,patch_text,delta=self.stream('repair',repair,question=PATCH_TASK,complete=True)
            self.check(regression.state.state_id==untouched,'Repair branch did not advance the regression branch')
            self.check(delta['source']==0 and 0<delta['target']<256,'Repair specialist processed only its new instruction')
            patch=extract_json(patch_text)
            self.set(patch=patch)
            del repair
            self.checkpoint('repair_complete','The repair specialist proposed a patch. The second branch will independently design the regression cases.')
            self.event('regression','Regression specialist: design independent tests from the same inherited context.')
            regression,cases_text,delta=self.stream('regression',regression,question=TEST_TASK,complete=True)
            self.check(delta['source']==0 and 0<delta['target']<256,'Regression specialist processed only its new instruction')
            cases=extract_json(cases_text)
            self.set(cases=cases.get('cases'))
            self.checkpoint('branches_complete','The patch and regression cases are ready. Run them against both versions of the code.')
            self.event('verify','Running generated regression cases before and after the model patch.')
            self.stage('verify','active')
            result=verify(self.repo,self.output,patch,cases)
            self.report['verification']=result
            self.check(not result['before']['passed'] and result['after']['passed'],'Generated tests fail before the patch and pass after it')
            self.stage('verify','done')
            self.set(result=result,status='done');self.event('done','READ ONCE. TRANSFER STATE. BRANCH INTO ACTION.')
            self.report['success']=True;self.save()
        except BaseException as exc:self.fail(exc)
        finally:self.active=False


def handler(app):
    static=Path(__file__).parent/'web'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,body,content='application/json',code=200):
            if isinstance(body,(dict,list)):body=json.dumps(body).encode()
            elif isinstance(body,str):body=body.encode()
            self.send_response(code);self.send_header('Content-Type',content+'; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path=='/api/state':return self.send(app.snapshot())
            if self.path=='/api/report':return self.send(app.report)
            names={'/':'index.html','/style.css':'style.css','/app.js':'app.js','/logo.png':'logo.png'}
            if self.path in names:
                name=names[self.path];kind={'html':'text/html','css':'text/css','js':'text/javascript','png':'image/png'}[name.split('.')[-1]]
                return self.send((static/name).read_bytes(),kind)
            self.send({'error':'Not found'},code=404)
        def do_POST(self):
            # Only the same-origin page may initiate GPU work.
            expected='http://'+self.headers.get('Host','')
            if self.headers.get('Origin')!=expected:return self.send({'error':'Origin rejected'},code=403)
            if self.path=='/api/start':return self.send({'started':app.start()})
            if self.path=='/api/continue':return self.send({'continued':app.advance()})
            self.send({'error':'Not found'},code=404)
    return Handler


def add_arguments(p):
    p.add_argument('--port',type=int,default=8768)
    p.add_argument('--max-tokens',type=int,default=3072)
    p.add_argument('--no-browser',action='store_true')
    p.add_argument('--headless',action='store_true',help='Run the real GPU demo once and exit, saving evidence')
    p.add_argument('--ui-only',action='store_true',help='Preview layout without loading models; Run is disabled')
    p.add_argument('--host', default='127.0.0.1', help='Use 0.0.0.0 inside Docker')
    p.add_argument('--model-manifest', type=Path, help='Explicit pinned external GGUF profile')


def run(args):
    if not 192<=args.max_tokens<=4096:
        raise ValueError('max-tokens must be 192..4096')
    if args.ui_only and args.headless:
        raise ValueError('--ui-only and --headless cannot be combined')
    from latentport.hub import bulk_home
    output=bulk_home()/'demos'/('transplant-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid()))
    output.mkdir(parents=True,exist_ok=False)
    app=App(output,args)
    url=f'http://127.0.0.1:{args.port}'
    print('LatentPort / Telepathy for LLMs')
    print('Evidence:',output)
    with quiet_native(output/'native.log') as console:
        try:
            if args.ui_only:app.set(status='preview',phase='ready',message='UI preview - launch without --ui-only to run the models')
            else:threading.Thread(target=app.prepare,daemon=True).start()
            if args.headless:
                while app.snapshot()['status'] not in ('done','error'):time.sleep(.25)
                final=app.snapshot();console.write(json.dumps({'status':final['status'],'message':final['message'],'speedup':final['speedup']})+'\n');console.flush()
                if final['status']=='error':raise RuntimeError(final['message'])
            else:
                server=ThreadingHTTPServer((args.host,args.port),handler(app))
                console.write('Open '+url+'\n');console.flush()
                if not args.no_browser:webbrowser.open(url)
                try:
                    server.serve_forever()
                finally:
                    server.server_close()
        except KeyboardInterrupt:
            return 0
        finally:
            if app.port:app.port.close()
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    return run(p.parse_args(argv))

if __name__=='__main__':main()
