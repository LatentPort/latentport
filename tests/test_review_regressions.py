"""Regression tests for review findings; no model weights or GPU required."""
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from threading import RLock
from dataclasses import replace
import torch
from latentport._native.engine import Engine,DecodeHandle
from latentport.gguf import GGUFBackend
from latentport._tokens import token_ids
from latentport.errors import CompatibilityError, CapabilityError
from latentport.identity import ModelIdentity
from latentport.state import State
from latentport.model import Model
from test_runtime import FakeBackend,FakeCodec,FakeHandle

def test_park_failure():
 e=Engine.__new__(Engine);e._lock=RLock();e._active={}
 old=DecodeHandle(e,'target',{'clock':1},2,False);old.dirty=True;e._active['target']=old
 def fail(_):raise RuntimeError('snapshot allocation failed')
 e._snapshot=fail
 try:e._park('target')
 except RuntimeError:pass
 assert e._active['target'] is old and old.dirty and old.clock == 2
 e._install=lambda *args: pytest.fail('must not reinstall a stale snapshot')
 old._activate()
 e._snapshot=lambda _: {'clock':2}
 e._park('target')
 assert 'target' not in e._active and old.values == {'clock':2} and not old.dirty

def test_codec_retry():
 backend=GGUFBackend.__new__(GGUFBackend);backend._codecs={};backend._side=lambda _: 'target'
 backend.manifest={'models':{'target':{}},'runtime':{'template_kwargs':{'target':{}}}}
 calls=[]
 def tokenize(side,text):calls.append(text);return [2]
 backend._native=lambda:SimpleNamespace(tokenize=tokenize)
 class Tokenizer:
  def __call__(self,*args,**kwargs):return {'input_ids':[1]}
 with patch('latentport.gguf.model_snapshot',return_value=Path('.')),patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
  try:backend.codec(None)
  except CompatibilityError:pass
  else:raise AssertionError('must reject first call')
  with pytest.raises(CompatibilityError): backend.codec(None)
  assert not backend._codecs and len(calls)==2

def test_native_integer_overflow():
 captured=[]
 e=SimpleNamespace(_lock=RLock(),_park=lambda side:None,_install=lambda *a:None,_ensure_room=lambda *a:None,
  _model=lambda side:(object(),None),binding=SimpleNamespace(check=lambda value:value),
  lib=SimpleNamespace(lp_prefill=lambda handle,ids,n:captured.extend(ids[:n]) or 0),_snapshot=lambda side:{})
 for ids in ([2**32+65], [2**31], [-1], [True], [1.5]):
  with pytest.raises(ValueError): Engine.advance(e,'target',{},1,ids)
 assert captured == []
 e._model=lambda side:(object(),SimpleNamespace(vocabulary=256))
 with pytest.raises(ValueError,match='vocabulary'): Engine.advance(e,'target',{},1,[256])
 Engine.advance(e,'target',{},1,[65])
 assert captured == [65]

@pytest.mark.parametrize("resume", ["same", "fork", "disk"])
def test_unicode_resumption(resume, tmp_path):
 class ByteCodec(FakeCodec):
  def decode(self,ids,*,special=False):return bytes(i for i in ids if special or i != 33).decode('utf-8',errors='replace')
 class ByteHandle(FakeHandle):
  schedule=(226,130,172,33)
  def __init__(self,clock):self.clock=clock;self.logits=self._logits(self.schedule[clock-1])
  def step(self,token):self.clock+=1;self.logits=self._logits(self.schedule[min(self.clock-1,3)])
 b=FakeBackend();b.codec=lambda _:ByteCodec();b.begin_decode=lambda ident,values,clock,**kw:ByteHandle(clock)
 ident=ModelIdentity('target','one','fake','test','tok','template',b.state_abi)
 parent=State(ident,1,'answer_ready','test',rendered_text='x',rendered_token_ids=(120,),messages_complete=False,_components={'H':torch.ones(1,dtype=torch.bfloat16)})
 session=Model(ident,b,role='target').session(parent)
 first=session.generate(max_new_tokens=1)
 assert first.text == ''
 if resume == 'fork': session=session.fork()
 if resume == 'disk': session=session.model.session(State.load(session.save(tmp_path/'state')))
 chunks=[first]+[session.generate(max_new_tokens=1) for _ in range(3)]
 assert ''.join(a.text for a in chunks) == '\u20ac'
 assert chunks[-1].finished

def test_compaction_weight_identity():
 from test_compact import state,GEO,compact
 parent=state(8192);parent=replace(parent,identity=replace(parent.identity,weights='bf16-sha256:unqualified-finetune'))
 with patch('latentport._compact.model_geometry',return_value=GEO):
  with pytest.raises(CapabilityError,match='qualification receipt'): compact(parent,force=True)
  child=compact(parent,force=True,allow_unqualified=True)
 assert child.physical_kv_length == 1024


def test_compaction_catalog_fingerprints_and_exact_weights():
 from hashlib import sha256
 from latentport._compact.compatibility import catalog, require_weights
 from test_compact import IDENTITY
 for repo, spec in catalog().items():
  digest=sha256()
  for name, receipt in sorted(spec['weight_files'].items()):
   digest.update(name.encode()+b'\0'+receipt['sha256'].encode())
  assert digest.hexdigest() == spec['safetensors_fingerprint']
  ident=replace(IDENTITY,repo_id=repo,revision=spec['revision'])
  for backend in ('bf16','nf4','reference'):
   known=replace(ident,backend=backend,weights=backend+'-sha256:'+digest.hexdigest()+':execution:test')
   require_weights(known)
   with pytest.raises(CapabilityError): require_weights(replace(known,weights='modified'))
  for fingerprint in spec['gguf_sha256']:
   require_weights(replace(ident,backend='gguf',weights='gguf-sha256:'+fingerprint))
  with pytest.raises(CapabilityError):
   require_weights(replace(ident,backend='gguf',weights='gguf-sha256:unknown'))
  with pytest.raises(CapabilityError,match='revision'):
   require_weights(replace(ident,revision='untrained'),allow_unqualified=True)
 with pytest.raises(TypeError): require_weights(IDENTITY,allow_unqualified=1)


def test_model_vocabulary_checked_before_append():
 backend=FakeBackend()
 codec=FakeCodec();codec.vocabulary_size=128
 backend.codec=lambda _:codec
 ident=ModelIdentity('target','one','fake','test','tok','template',backend.state_abi)
 model=Model(ident,backend,role='target')
 parent=model.capture('x')
 before=list(backend.calls)
 with pytest.raises(ValueError,match='vocabulary'): model.append(parent,[128])
 assert backend.calls == before
