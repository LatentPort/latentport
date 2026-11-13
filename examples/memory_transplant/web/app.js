const $=id=>document.getElementById(id);
let state=null,scene=0,liveScene=0,lastPhase='',selected='native',branch='repair',requestPending=false;
const fmt=x=>x==null?'\u2014':x.toFixed(3)+' s';
const phaseScene=p=>['native','native_complete','read','transfer','generate','composite_complete'].includes(p)?1:['fork','repair','repair_complete','regression','branches_complete'].includes(p)?2:['verify','done'].includes(p)?3:0;
const headings=[
 ['LATENTPORT / FULL-STATE TRANSFER','Telepathy<br><em>for LLMs.</em>','One model reads. Another inherits its internal working context.'],
 ['CROSS-MODEL CONTEXT TRANSFER','Move the state.<br><em>Skip the prompt replay.</em>','LatentPort translates 4B internal state into state 27B can use.'],
 ['PROGRAMMABLE MODEL STATE','Fork the context.<br><em>Multiply the possibilities.</em>','Two independent 27B sessions inherit the same prepared context.'],
 ['TRANSFER. REUSE. CONTINUE.','Telepathy for LLMs.<br><em>Measured. Demonstrated.</em>','Internal state crosses models. The original prompt is not processed by 27B.']
];
function showScene(index){scene=index;for(const el of document.querySelectorAll('[data-screen]'))el.hidden=Number(el.dataset.screen)!==index;for(const b of document.querySelectorAll('[data-scene]')){b.classList.toggle('selected',Number(b.dataset.scene)===index);b.setAttribute('aria-current',Number(b.dataset.scene)===index?'step':'false');b.disabled=Number(b.dataset.scene)>liveScene;}const [eye,title,desc]=headings[index];$('eyebrow').textContent=eye;$('heading').innerHTML=title;$('explanation').textContent=desc;}
function stream(id,run,override){const el=$(id);let text=override??run.text;if(override==null&&run.status==='done'&&text.includes('</think>'))text=text.split('</think>').at(-1);text=text.replace(/<\|im_end\|>/g,'').replace(/<\/?think>/g,'').trim();if(text&&el.textContent!==text){const atBottom=el.scrollHeight-el.scrollTop-el.clientHeight<60;el.textContent=text;if(atBottom)el.scrollTop=el.scrollHeight;}el.classList.toggle('live',run.status==='running');}
function nextAction(s){if(scene!==liveScene)return ['Return to live step','Reviewing an earlier chapter.','review'];if(s.status==='error')return ['Run stopped',s.message,'error'];if(s.status==='ready')return ['Run native 27B \u2192','First measure the normal route: 27B processes the whole prompt.','start'];if(s.status==='paused'){const map={native_complete:['Try LatentPort \u2192','Now transfer internal state instead of feeding the prompt to 27B.'],composite_complete:['Fork this context \u2192','Reuse the inherited context in two independent 27B sessions.'],repair_complete:['Run test engineer \u2192','The first branch has finished. Continue the second from the same starting state.'],branches_complete:['Apply & test \u2192','Verify that the inherited context produced useful output.']};return [...(map[s.phase]||['Continue \u2192','Continue the demonstration.']),'continue'];}if(s.status==='done')return ['View run evidence \u2197','Verified by executing the generated tests.','report'];if(s.status==='preview')return ['Layout preview','Start without --ui-only to run the models.','preview'];return [s.status==='loading'?'Preparing GPU...':'Working...',s.status==='loading'?'Loading and warming both models.':s.message,'wait'];}
function render(s){state=s;document.body.classList.toggle('error',s.status==='error');const nextScene=phaseScene(s.phase);if(s.phase!==lastPhase){if(nextScene!==liveScene)scene=nextScene;liveScene=nextScene;if(s.phase==='native')selected='native';if(['read','transfer','generate','composite_complete'].includes(s.phase))selected='composite';if(['fork','repair','repair_complete'].includes(s.phase))branch='repair';if(['regression','branches_complete'].includes(s.phase))branch='regression';lastPhase=s.phase;queueNarration(s.phase,s);}showScene(scene);
$('repoMeta').textContent=s.context_tokens?'Demo input: '+s.context_tokens.toLocaleString()+' tokens of code. Task: diagnose a checkout bug.':'Preparing the repository and GPU...';
const isNative=['native','native_complete'].includes(s.phase);$('baselineFlow').hidden=!isNative;$('compositeFlow').hidden=isNative;if(scene===1){$('heading').innerHTML=isNative?'Normally, 27B<br><em>reads the whole prompt.</em>':'With LatentPort,<br><em>27B receives state.</em>';$('explanation').textContent=isNative?'Baseline: full-prompt processing on 27B.':'4B reads. LatentPort translates its state for 27B.';}$('flowLabel').textContent=isNative?'BASELINE: THE PROMPT GOES TO 27B':'LATENTPORT: INTERNAL STATE GOES TO 27B';
for(const k of ['read','transfer','generate']){const el=document.querySelector(`[data-node="${k}"]`);el.className='flow-node'+(k==='transfer'?' bridge':'')+' '+s.stages[k];}
$('wire1').classList.toggle('active',s.stages.transfer==='active');$('wire2').classList.toggle('active',s.stages.transfer==='active');
const descriptions={native:'27B is processing every token of the original prompt.',native_complete:'Baseline captured. Now skip this large-model prompt processing.',read:'4B is turning the prompt into its internal working state.',transfer:'LatentPort translates that state across the two different models.',generate:'27B generates from the transferred state, without prompt replay.',composite_complete:'Verified: 27B processed zero original-prompt tokens.'};
$('stepDescription').textContent=descriptions[s.phase]||'4B reads the prompt. The bridge translates state. 27B generates.';
for(const b of document.querySelectorAll('[data-stream]')){b.classList.toggle('selected',b.dataset.stream===selected);b.setAttribute('aria-pressed',String(b.dataset.stream===selected));}
stream('diagnosis',s.runs[selected]);$('diagTokens').textContent=s.runs[selected].tokens+' tokens';$('diagLabel').textContent=s.runs[selected].status==='done'?'MODEL RESPONSE':'LIVE MODEL OUTPUT';
const ratio=s.speedup||(s.runs.native.ttft&&s.runs.composite.ttft?s.runs.native.ttft/s.runs.composite.ttft:null);
$('speedup').textContent=ratio?ratio.toFixed(2)+(ratio>=1?'\u00d7 faster first token':'\u00d7 TTFT ratio'):'Time to first token';$('proofCount').textContent=s.checks.some(c=>c.startsWith('27B prompt tokens'))?'0 original-prompt tokens processed by 27B':'Same prompt / sequential runs';
for(const key of ['repair','regression']){$(key+'Status').textContent=({waiting:'Waiting',running:'Generating...',done:'Complete'})[s.runs[key].status];}
for(const b of document.querySelectorAll('[data-branch]')){b.classList.toggle('selected',b.dataset.branch===branch);b.setAttribute('aria-pressed',String(b.dataset.branch===branch));}
const r=s.runs[branch];$('branchName').textContent=branch==='repair'?'Repair engineer':'Test engineer';$('branchTask').textContent=branch==='repair'?'Replace the faulty assignment with the correct conversion.':'Design checkout cases for USD, JPY and KWD.';$('branchTokens').textContent=r.tokens+' tokens';$('branchTime').textContent='First token: '+fmt(r.ttft);$('branchLabel').textContent=r.status==='done'?'GENERATED ARTIFACT':'LIVE MODEL OUTPUT';let artifact;
if(branch==='repair'&&s.patch)artifact=s.patch.file+'\n\n\u2212 '+s.patch.old+'\n+ '+s.patch.new;
const cases=s.cases||s.result?.cases;if(branch==='regression'&&Array.isArray(cases))artifact=cases.map(c=>c.currency+' | '+c.quantity+' x '+c.price+'\nExpected: '+c.expected_minor+' minor units').join('\n\n');
if(r.status==='waiting')$('branchOutput').textContent='This branch starts from the same inherited context. Waiting for its instruction.';else stream('branchOutput',r,artifact);
if(s.patch)$('patchDiff').textContent='\u2212 '+s.patch.old+'\n+ '+s.patch.new;
$('verifyBadge').textContent=s.result?'VERIFIED':'RUNNING TESTS';$('verifyBadge').className='badge'+(s.result?' pass':'');
if(s.result){$('beforeChip').className='fail';$('beforeChip').querySelector('b').textContent='FAIL';$('afterChip').className='pass';$('afterChip').querySelector('b').textContent=s.result.after.test_count?s.result.after.test_count+' PASS':'PASS';$('resultExplanation').textContent='Generated cases fail on the original code. The full suite passes after the patch.';}
$('finalRatio').textContent=ratio?ratio.toFixed(2)+'\u00d7':'\u2014';$('finalRatioLabel').textContent=ratio&&ratio<1?'TTFT ratio':'faster first token';
$('finalPromptCount').textContent=s.checks.some(c=>c.startsWith('27B prompt tokens'))?'0':'\u2014';$('finalBranches').textContent=s.runs.repair.status==='done'&&s.runs.regression.status==='done'?'2':'\u2014';
const [label,hint,action]=nextAction(s);$('action').textContent=label;$('action').dataset.action=action;$('action').disabled=requestPending||['wait','error','preview'].includes(action);$('nextHint').textContent=hint;$('status').textContent=({ready:'READY TO BEGIN',paused:'YOUR NEXT STEP',done:'RUN VERIFIED',running:'LIVE INFERENCE',loading:'PREPARING GPU',error:'RUN STOPPED',preview:'LAYOUT PREVIEW'})[s.status];
}
function animate(){if(state){const times={};for(const key of ['native','composite']){const r=state.runs[key];times[key]=r.ttft??(r.status==='running'?Math.max(0,Date.now()/1000-r.started):null);}const scale=Math.max(8,...Object.values(times).map(x=>x||0));for(const key of ['native','composite']){$(key+'Time').textContent=fmt(times[key]);$(key+'Bar').style.width=((times[key]||0)/scale*100)+'%';}}requestAnimationFrame(animate);}animate();
async function poll(){try{const response=await fetch('/api/state');if(!response.ok)throw Error('Connection lost');render(await response.json());}catch(error){$('status').textContent='CONNECTION LOST';$('nextHint').textContent='Keep the demo server running in PowerShell.';$('action').disabled=true;}setTimeout(poll,150);}poll();
$('action').onclick=async()=>{const action=$('action').dataset.action;if(action==='review'){showScene(liveScene);render(state);return;}if(action==='report'){window.open('/api/report','_blank','noopener');return;}requestPending=true;$('action').disabled=true;try{const response=await fetch(action==='start'?'/api/start':'/api/continue',{method:'POST'});if(!response.ok)throw Error('Could not advance the run.');scene=liveScene;}catch(error){$('nextHint').textContent=error.message;}finally{requestPending=false;}};
for(const b of document.querySelectorAll('[data-scene]'))b.onclick=()=>{showScene(Number(b.dataset.scene));if(state)render(state);};
for(const b of document.querySelectorAll('[data-stream]'))b.onclick=()=>{selected=b.dataset.stream;if(state)render(state);};
for(const b of document.querySelectorAll('[data-branch]'))b.onclick=()=>{branch=b.dataset.branch;if(state)render(state);};
for(const b of document.querySelectorAll('[data-latest]'))b.onclick=()=>{const el=$(b.dataset.latest);el.scrollTop=el.scrollHeight;};
$('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen().catch(()=>{});};
$('source').onclick=()=>{const files=state?.files||[];$('fileList').replaceChildren();for(const f of files){const b=document.createElement('button');b.textContent=f.name;b.onclick=()=>{for(const other of $('fileList').children)other.classList.remove('selected');b.classList.add('selected');$('fileText').textContent=f.text;};$('fileList').append(b);}if($('fileList').children.length)$('fileList').children[0].click();$('repoDialog').showModal();};$('closeModal').onclick=()=>$('repoDialog').close();

// Presenter narration follows real server events. It never changes model timings.
const narrationQueue=[];
let narrating=false;
const narrationCopy={
 loading:'LatentPort connects two models through their internal state.',
 warming:'Both models are getting ready on this GPU.',
 ready:'Telepathy for LLMs. One model reads. Another inherits the context.',
 native:'Normally, the large model must process the entire prompt itself.',
 native_complete:'LatentPort changes who does that reading. Watch the handoff.',
 read:'4B reads the prompt. LatentPort translates its state. 27B inherits the context.',
 composite_complete:'Zero original-prompt tokens processed by 27B. That is the handoff.',
 fork:'The inherited context forks into two sessions. The first takes a new task.',
 repair_complete:'The other branch still has its own untouched starting state.',
 regression:'The second branch gets a different task. The history is not replayed.',
 branches_complete:'One state transfer supplied context for both branches.',
 done:'Transfer context across models. Fork it. Continue. Telepathy for LLMs.'
};
function queueNarration(phase,s){let caption=narrationCopy[phase];if(s.status==='error')caption='The run stopped. The error is shown below.';if(s.status==='preview')caption='Telepathy for LLMs. Internal state moves between models.';if(!caption)return;
// Rapid native phases share one complete explanation; keep only the newest pending
// caption so subtitles remain in step with what is on screen.
narrationQueue.splice(0,narrationQueue.length,caption);if(!narrating)playNarration();}
function playNarration(){const caption=narrationQueue.shift();if(!caption){narrating=false;return;}narrating=true;const el=$('narration');el.textContent='';el.classList.add('streaming');const words=caption.split(' ');let i=0;const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
function tick(){i=reduced?words.length:i+1;el.textContent=words.slice(0,i).join(' ');if(i<words.length)setTimeout(tick,210);else{el.classList.remove('streaming');$('narrationAccessible').textContent=caption;setTimeout(playNarration,4000);}}tick();}
