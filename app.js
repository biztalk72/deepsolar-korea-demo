import { initializeOfficialView, loadOfficialResults, updateOfficialScope } from './real-sample-view.js';
import { SIZE, DATASET_VERSION, createDataset, degradeImage, predictMask, fingerprintDataset } from './ml-core.js';

const $ = id => document.getElementById(id);
const $$ = selector => [...document.querySelectorAll(selector)];
const dataset = createDataset(72);
const originalMasks = new Map(dataset.tiles.map(tile => [tile.id, new Uint8Array(tile.mask)]));
const LABEL_KEY = 'deepsolar-browser-labels-v1';
const RUN_KEY = 'deepsolar-browser-runs-v1';
const splitNames = {train:'학습', validation:'검증', test:'평가'};
const routeNames = {overview:'대시보드',review:'데이터 · 라벨 검수',experiments:'해상도 · 학습 실험',results:'모델 평가 · 비교',official:'공식 샘플 학습 결과'};
const pendingMessages = [];
let currentTileId = dataset.tiles[0].id;
let splitFilter = 'all';
let brushValue = 1;
let brushSize = 3;
let factor = 1;
let cursor = {x:32,y:32,visible:false};
let painting = false;
let lastPoint = null;
let worker = null;
let activeRun = null;
let progressHistory = [];
let dirtyTiles = new Set();
let toastTimer;
let runs = [];
let selectedRunId = null;

function notify(message) {
  $('toast').textContent = message;
  $('toast').classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').classList.remove('show'), 4200);
}
function pct(value) { return Number.isFinite(value) ? (value*100).toFixed(1)+'%' : '—'; }
function num(value,digits=4) { return Number.isFinite(value) ? value.toFixed(digits) : '—'; }
function currentTile() { return dataset.tiles.find(t => t.id === currentTileId) || dataset.tiles[0]; }
function canEdit() { return currentTile().split === 'train' && !activeRun; }
function fingerprint() { return fingerprintDataset(dataset.tiles); }
function validMetric(m) { return m && ['f1','iou','precision','recall'].every(k => Number.isFinite(m[k]) && m[k]>=0 && m[k]<=1); }
function validStoredRun(run) {
  if (!run || typeof run.id!=='string' || run.id.length>100 || typeof run.name!=='string' || run.name.length>80 || typeof run.fingerprint!=='string' || !/^fnv1a-[a-f0-9]{8}$/.test(run.fingerprint) || !Number.isFinite(Date.parse(run.createdAt))) return false;
  if (run.model?.format!=='deepsolar-browser-logistic-v1' || !['weights','featureMean','featureStd'].every(k => Array.isArray(run.model[k]) && run.model[k].length===9 && run.model[k].every(v=>Number.isFinite(v)&&Math.abs(v)<1e6)) || !run.model.featureStd.every(v=>v>=1e-6)) return false;
  if (![1,2,4,8].includes(run.options?.factor) || !Number.isInteger(run.options?.epochs) || run.options.epochs<1 || run.options.epochs>300 || !Number.isFinite(run.options.learningRate) || run.options.learningRate<=0 || run.options.learningRate>1) return false;
  if (!Number.isFinite(run.elapsedMs) || run.elapsedMs<0 || !Number.isInteger(run.trainingPixels) || run.trainingPixels<1 || run.trainingPixels>16*SIZE*SIZE || ![run.metrics?.train,run.metrics?.validation,run.metrics?.test].every(validMetric)) return false;
  if (!Array.isArray(run.history) || !run.history.length || run.history.length>301 || !run.history.every(p=>p && Number.isInteger(p.epoch) && p.epoch>=1 && p.epoch<=run.options.epochs && Number.isFinite(p.loss) && p.loss>=0 && p.loss<1e6 && validMetric(p.val))) return false;
  return true;
}
function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(LABEL_KEY) || 'null');
    if (saved && saved.schema === 1 && saved.datasetId === dataset.id) {
      let restored = 0;
      for (const tile of dataset.tiles.filter(t => t.split === 'train')) {
        const mask = saved.labels?.[tile.id];
        if (Array.isArray(mask) && mask.length === SIZE*SIZE && mask.every(v => v===0 || v===1 || v===255)) {
          tile.mask = Uint8Array.from(mask); restored++;
        }
      }
      if (restored && restored !== 16) pendingMessages.push('저장된 라벨 중 유효한 학습 라벨만 복원했습니다.');
    } else if (saved) pendingMessages.push('현재 데이터 버전과 다른 저장 라벨은 불러오지 않았습니다.');
  } catch { pendingMessages.push('브라우저 저장 라벨을 불러올 수 없어 기본 데이터로 시작했습니다.'); }
  try {
    const saved = JSON.parse(localStorage.getItem(RUN_KEY) || 'null');
    if (saved?.schema === 1 && saved.datasetId === dataset.id && Array.isArray(saved.runs)) {
      runs = saved.runs.filter(validStoredRun).slice(0,12);
      if (runs.length < Math.min(saved.runs.length,12)) pendingMessages.push('형식이 올바르지 않은 저장 실험은 불러오지 않았습니다.');
    }
  } catch { pendingMessages.push('브라우저 실험 기록을 불러올 수 없어 새 목록을 시작했습니다.'); }
  selectedRunId = runs[0]?.id || null;
}
function saveLabels(showNotice=true) {
  const labels = Object.fromEntries(dataset.tiles.filter(t=>t.split==='train').map(t=>[t.id,Array.from(t.mask)]));
  try {
    localStorage.setItem(LABEL_KEY, JSON.stringify({schema:1,datasetId:dataset.id,synthetic:true,savedAt:new Date().toISOString(),labels}));
    dirtyTiles.clear();
    renderReviewMeta();
    if (showNotice) notify('학습 라벨을 이 브라우저에 저장했습니다.');
    return true;
  } catch {
    if (showNotice) notify('브라우저 저장 공간을 사용할 수 없습니다. 라벨 내보내기로 보관하세요.');
    return false;
  }
}
function saveRuns() {
  try { localStorage.setItem(RUN_KEY,JSON.stringify({schema:1,datasetId:dataset.id,runs})); return true; }
  catch { notify('실험은 완료했지만 브라우저 저장에 실패했습니다. 결과를 내보내세요.'); return false; }
}
function download(data,filename) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function exportLabels() {
  download({schema:'deepsolar-browser-review-v1',datasetId:dataset.id,datasetVersion:DATASET_VERSION,seed:dataset.seed,synthetic:true,fingerprint:fingerprint(),exportedAt:new Date().toISOString(),labelValues:{background:0,solar:1,ignore:255},note:'합성 학습 타일만 포함합니다. 실제 위성 영상 라벨이 아닙니다.',tiles:dataset.tiles.filter(t=>t.split==='train').map(t=>({id:t.id,name:t.name,split:t.split,width:t.width,height:t.height,mask:Array.from(t.mask)}))},'deepsolar-synthetic-labels.json');
  notify('합성 학습 라벨 JSON을 내보냈습니다.');
}
function exportResults() {
  if (!runs.length) return;
  download({schema:'deepsolar-browser-experiments-v1',datasetId:dataset.id,synthetic:true,exportedAt:new Date().toISOString(),notice:'브라우저 로지스틱 기준선의 합성 데이터 결과입니다. 한국 실제 위성 탐지 정확도, 발전량 또는 독립 출시 평가를 의미하지 않습니다.',runs},'deepsolar-browser-experiments.json');
  notify('모델과 실제 평가 지표가 포함된 실험 결과를 내보냈습니다.');
}
function imageCanvas(rgb,width,height,mask=null,color=[91,222,147],alpha=.48) {
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  const ctx=canvas.getContext('2d');const pixels=ctx.createImageData(width,height);
  for(let i=0;i<width*height;i++) {
    let r=rgb[i*3],g=rgb[i*3+1],b=rgb[i*3+2];
    if(mask && (mask[i]===1 || mask[i]===255)) {
      const c=mask[i]===255?[233,184,82]:color;
      const a=mask[i]===255?.65:alpha;
      r=r*(1-a)+c[0]*a;g=g*(1-a)+c[1]*a;b=b*(1-a)+c[2]*a;
    }
    pixels.data.set([r,g,b,255],i*4);
  }
  ctx.putImageData(pixels,0,0);return canvas;
}
function drawRGB(canvas,rgb,width=SIZE,height=SIZE,mask=null,color=[91,222,147],alpha=.48) {
  const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(imageCanvas(rgb,width,height,mask,color,alpha),0,0,canvas.width,canvas.height);
}
function renderHero() {
  const canvas=$('hero-canvas'),ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;
  ctx.fillStyle='#aebd9c';ctx.fillRect(0,0,canvas.width,canvas.height);
  const selected=[dataset.tiles[0],dataset.tiles[4],dataset.tiles[9],dataset.tiles[13],dataset.tiles[18],dataset.tiles[22]];
  const tileW=164,tileH=164;
  selected.forEach((t,i)=>{
    const x=(i%3)*174-10,y=Math.floor(i/3)*174-4;
    ctx.drawImage(imageCanvas(t.rgb,SIZE,SIZE),x,y,tileW,tileH);
    ctx.strokeStyle='#d8e4ce66';ctx.lineWidth=1;ctx.strokeRect(x+12,y+12,tileW-24,tileH-24);
    ctx.fillStyle='#dcebdc';ctx.font='8px monospace';ctx.fillText(String(i+1).padStart(2,'0'),x+17,y+24);
  });
  ctx.strokeStyle='#e5efd326';ctx.lineWidth=.7;
  for(let x=0;x<canvas.width;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();}
  for(let y=0;y<canvas.height;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();}
}
function route() {
  const name=location.hash.slice(1);
  if(name==='main'){$('main').focus();return;}
  const next=Object.hasOwn(routeNames,name)?name:'overview';
  $$('.view').forEach(el=>{const active=el.id==='view-'+next;el.hidden=!active;el.classList.toggle('active',active);});
  $$('[data-route]').forEach(el=>{const active=el.dataset.route===next;el.classList.toggle('active',active);if(active)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
  $('breadcrumb-current').textContent=routeNames[next];document.title=routeNames[next]+' · DeepSolar Korea';
  if(next==='review')renderReview();
  if(next==='experiments'){renderResolution();renderFingerprint();}
  if(next==='results')renderRuns();
  updateOfficialScope(next);
  if(next==='official')loadOfficialResults();
  window.scrollTo({top:0,behavior:'instant'});
}
function renderStats() {
  $('stat-runs').replaceChildren(document.createTextNode(String(runs.length)));
  const small=document.createElement('small');small.textContent='회';$('stat-runs').append(small);
  $('stat-runs-note').textContent=runs.length?'최근 '+runs.length+'개 실험을 이 브라우저에 보관':'브라우저에서 첫 실험을 시작하세요';
  $('stat-f1').textContent=runs.length?pct(runs[0].metrics.test.f1):'—';
  $('stat-f1-note').textContent=runs.length?'고정 합성 평가 타일 · 실제 연산 결과':'학습 완료 후 실제 측정값 표시';
}
function renderTiles() {
  const grid=$('tile-grid');grid.replaceChildren();
  const visible=dataset.tiles.filter(t=>splitFilter==='all'||t.split===splitFilter);
  if(!visible.some(t=>t.id===currentTileId))currentTileId=visible[0]?.id||dataset.tiles[0].id;
  for(const tile of visible) {
    const button=document.createElement('button');button.className='tile-card';button.classList.toggle('selected',tile.id===currentTileId);button.type='button';button.dataset.tileId=tile.id;button.setAttribute('aria-pressed',String(tile.id===currentTileId));button.setAttribute('aria-label',tile.name+' · '+splitNames[tile.split]);
    const canvas=document.createElement('canvas');canvas.width=SIZE;canvas.height=SIZE;canvas.setAttribute('aria-hidden','true');drawRGB(canvas,tile.rgb);
    const info=document.createElement('span');info.className='tile-card-info';
    const title=document.createElement('strong');title.textContent=tile.name;
    const detail=document.createElement('small');detail.textContent=String(tile.id);const split=document.createElement('i');split.textContent=splitNames[tile.split];detail.append(split);
    info.append(title,detail);button.append(canvas,info);button.addEventListener('click',()=>{currentTileId=tile.id;cursor.visible=false;renderReview();});grid.append(button);
  }
}
function drawEditor() {
  const tile=currentTile();const canvas=$('editor-canvas');
  drawRGB(canvas,tile.rgb,SIZE,SIZE,$('show-overlay').checked?tile.mask:null);
  if(cursor.visible&&canEdit()) {
    const ctx=canvas.getContext('2d');const s=canvas.width/SIZE;
    const r=Math.floor(brushSize/2);const x=Math.max(0,cursor.x-r),y=Math.max(0,cursor.y-r);
    ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(x*s+.5,y*s+.5,Math.min(brushSize,SIZE-x)*s-1,Math.min(brushSize,SIZE-y)*s-1);
    ctx.strokeStyle='#102f30';ctx.lineWidth=.8;ctx.strokeRect(x*s+2,y*s+2,Math.min(brushSize,SIZE-x)*s-4,Math.min(brushSize,SIZE-y)*s-4);
  }
}
function renderReviewMeta() {
  const tile=currentTile(),editable=canEdit();
  $('tile-name').textContent=tile.name;$('tile-scene').textContent=tile.scene+' · 합성 영상';
  $('tile-split').textContent=splitNames[tile.split];$('tile-split').className='badge '+tile.split;
  $('editor-canvas').classList.toggle('readonly',!editable);
  $('editor-canvas').setAttribute('aria-label',editable?'학습 라벨 편집 영상. 방향키로 커서를 이동하고 Space 키로 칠할 수 있습니다.':'읽기 전용 '+splitNames[tile.split]+' 영상');
  $('brush-tools').hidden=!editable;$('readonly-note').hidden=editable;
  $('readonly-note').querySelector('span').textContent=activeRun?'학습 중에는 라벨 편집이 잠깐 잠깁니다.':'검증·평가 데이터의 라벨은 수정할 수 없습니다.';
  $('reset-mask').disabled=!editable;$('save-labels').disabled=!dirtyTiles.size||!!activeRun;
  const fg=tile.mask.reduce((n,v)=>n+(v===1),0),ignored=tile.mask.reduce((n,v)=>n+(v===255),0);
  $('label-pixels').textContent='태양광 '+fg+' px · 제외 '+ignored+' px';
  const status=$('label-save-status');status.classList.toggle('dirty',dirtyTiles.size>0);
  status.textContent=dirtyTiles.size?dirtyTiles.size+'개 타일 변경 · 저장 필요':'변경 사항 저장됨';
}
function renderReview() {renderTiles();renderReviewMeta();drawEditor();}
function markDirty(){dirtyTiles.add(currentTileId);renderReviewMeta();renderFingerprint();}
function stamp(x,y){
  if(!canEdit())return;
  const tile=currentTile(),r=Math.floor(brushSize/2);
  for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){
    const px=x+dx,py=y+dy;
    if(px>=0&&px<SIZE&&py>=0&&py<SIZE&&dx*dx+dy*dy<=r*r+1)tile.mask[py*SIZE+px]=brushValue;
  }
  $('show-overlay').checked=true;markDirty();
}
function pointerPosition(event){const rect=$('editor-canvas').getBoundingClientRect();return{x:Math.max(0,Math.min(SIZE-1,Math.floor((event.clientX-rect.left)/rect.width*SIZE))),y:Math.max(0,Math.min(SIZE-1,Math.floor((event.clientY-rect.top)/rect.height*SIZE)))};}
function paintLine(a,b){const distance=Math.max(Math.abs(b.x-a.x),Math.abs(b.y-a.y));for(let i=0;i<=distance;i++){const t=distance?i/distance:0;stamp(Math.round(a.x+(b.x-a.x)*t),Math.round(a.y+(b.y-a.y)*t));}}
function renderFingerprint(){const fp=fingerprint();$('training-fingerprint').textContent=fp;$('training-fingerprint').title=fp;}
function setupTileOptions(){
  const experiment=$('experiment-tile'),prediction=$('prediction-tile');
  for(const tile of dataset.tiles){const option=document.createElement('option');option.value=tile.id;option.textContent=tile.name+' · '+splitNames[tile.split];experiment.append(option);if(tile.split==='test')prediction.append(option.cloneNode(true));}
  experiment.value=dataset.tiles.find(t=>t.split==='validation').id;
}
function renderResolution(){
  const tile=dataset.tiles.find(t=>t.id===$('experiment-tile').value)||dataset.tiles[0];
  drawRGB($('resolution-original'),tile.rgb);drawRGB($('resolution-degraded'),degradeImage(tile.rgb,SIZE,SIZE,factor));
  $('factor-image-badge').textContent='×'+factor+' INPUT';
  $('factor-image-caption').textContent=factor===1?'원본 입력':factor+'배 축소 후 확대';
  $('factor-image-detail').textContent=(SIZE/factor)+' × '+(SIZE/factor)+' → 64 × 64';
  $('training-factor').textContent=factor===1?'원본 ×1':'축소 배율 ×'+factor;
}
function setRunning(running){
  $('start-training').disabled=running;$('stop-training').hidden=!running;
  $('epochs').disabled=running;$('run-name').disabled=running;
  $$('[data-factor]').forEach(b=>b.disabled=running);
  renderReviewMeta();
}
function renderLoss(history){
  const points=history.filter(p=>Number.isFinite(p.loss));const chart=$('loss-chart');
  if(!points.length){chart.replaceChildren();$('loss-empty').hidden=false;return;}
  $('loss-empty').hidden=true;
  const max=Math.max(.1,...points.map(p=>p.loss))*1.08,min=0,left=43,right=530,top=13,bottom=126;
  const epochMax=Math.max(1,points.at(-1).epoch);
  const coords=points.map(p=>[left+(p.epoch/epochMax)*(right-left),bottom-((p.loss-min)/(max-min))*(bottom-top)]);
  const path=coords.map((p,i)=>(i?'L':'M')+p.map(v=>v.toFixed(2)).join(',')).join(' ');
  const area=path+' L'+coords.at(-1)[0].toFixed(2)+','+bottom+' L'+coords[0][0].toFixed(2)+','+bottom+' Z';
  const yMid=(bottom+top)/2;
  chart.innerHTML='<path d="M43 '+top+'H530 M43 '+yMid+'H530 M43 '+bottom+'H530" stroke="#e5eddd" stroke-width="1"/><path d="'+area+'" fill="#edf4e5" stroke="none"/><path d="'+path+'" stroke="#73995a" fill="none" stroke-width="2"/><text x="4" y="18" fill="#a2b591" stroke="none" font-size="9">'+max.toFixed(2)+'</text><text x="4" y="'+(yMid+3)+'" fill="#a2b591" stroke="none" font-size="9">'+(max/2).toFixed(2)+'</text><text x="4" y="129" fill="#a2b591" stroke="none" font-size="9">0.00</text><text x="43" y="146" fill="#a2b591" stroke="none" font-size="8">0 epoch</text><text x="530" y="146" text-anchor="end" fill="#a2b591" stroke="none" font-size="8">'+epochMax+' epoch</text>';
}
function handleProgress(data){
  progressHistory.push(data);const percentage=Math.round(data.epoch/data.epochs*100);
  $('training-progress').value=percentage;$('training-percent').textContent=percentage+'%';
  $('progress-epoch').textContent=data.epoch+' / '+data.epochs;$('progress-loss').textContent=num(data.loss);$('progress-f1').textContent=pct(data.val?.f1);
  $('training-stage').textContent='학습 가중치 업데이트 중';
  $('training-log').textContent='Epoch '+data.epoch+' · 손실 '+num(data.loss)+' · 검증 F1 '+pct(data.val?.f1);
  renderLoss(progressHistory);
}
function trainingError(message){
  worker?.terminate();worker=null;activeRun=null;setRunning(false);
  $('training-status').textContent='실행 오류';$('training-stage').textContent='학습을 완료하지 못했습니다';$('training-log').textContent=message;
  notify('학습 오류: '+message);
}
function startTraining(event){
  event.preventDefault();if(activeRun)return;
  const epochs=Number($('epochs').value);const name=$('run-name').value.trim()||'실험 '+(runs.length+1);
  const options={epochs,learningRate:.2,factor};
  const id='run-'+Date.now()+'-'+Math.random().toString(16).slice(2,8);
  activeRun={id,name,fingerprint:fingerprint(),options,createdAt:new Date().toISOString()};
  if(dirtyTiles.size&&!saveLabels(false))notify('현재 라벨로 학습합니다. 저장 공간 제한이 있으니 라벨을 내보내세요.');
  progressHistory=[];setRunning(true);$('training-status').textContent='연산 중';$('training-status').className='badge';
  $('training-stage').textContent='학습 영상의 특징을 계산하고 있습니다';$('training-percent').textContent='0%';$('training-progress').value=0;
  $('progress-epoch').textContent='0 / '+epochs;$('progress-loss').textContent='—';$('progress-f1').textContent='—';
  $('training-log').textContent='학습 16 · 검증 4 · 평가 4 타일 / 축소 배율 ×'+factor;
  $('completion-callout').hidden=true;renderLoss([]);
  try {
    worker=new Worker(new URL('./training-worker.js',import.meta.url),{type:'module'});
    worker.addEventListener('error',event=>{if(activeRun?.id===id)trainingError(event.message||'브라우저에서 학습 작업을 실행할 수 없습니다.');});
    worker.addEventListener('message',event=>{
      const data=event.data;if(!activeRun||data.runId!==activeRun.id)return;
      if(data.type==='progress')handleProgress(data);
      if(data.type==='error')trainingError(data.message||'입력 데이터를 확인해주세요.');
      if(data.type==='complete'){
        const run={...activeRun,...data.result,id:activeRun.id,name:activeRun.name,fingerprint:activeRun.fingerprint,createdAt:activeRun.createdAt};
        runs.unshift(run);runs=runs.slice(0,12);selectedRunId=run.id;saveRuns();
        worker.terminate();worker=null;activeRun=null;setRunning(false);
        $('training-status').textContent='학습 완료';$('training-status').className='badge';
        $('training-stage').textContent='학습 및 분리 데이터 평가 완료';$('training-percent').textContent='100%';$('training-progress').value=100;
        $('progress-epoch').textContent=run.options.epochs+' / '+run.options.epochs;
        if(run.history?.length){const last=run.history.at(-1);$('progress-loss').textContent=num(last.loss);renderLoss(run.history);}
        $('progress-f1').textContent=pct(run.metrics.validation.f1);
        $('training-log').textContent='완료 · '+(run.elapsedMs/1000).toFixed(2)+'초 · 학습 픽셀 '+run.trainingPixels.toLocaleString()+' · Test F1 '+pct(run.metrics.test.f1);
        $('completion-callout').hidden=false;renderStats();renderRuns();
        $('run-name').value='실험 '+(runs.length+1);notify('실제 학습과 평가가 완료되었습니다. 모델 평가에서 결과를 확인하세요.');
      }
    });
    worker.postMessage({type:'train',runId:id,tiles:dataset.tiles,options});
  } catch(error){trainingError(error.message);}
}
function stopTraining(){
  if(!activeRun)return;worker?.terminate();worker=null;activeRun=null;setRunning(false);
  $('training-status').textContent='중지됨';$('training-status').className='badge muted';$('training-stage').textContent='사용자가 학습을 중지했습니다';
  $('training-log').textContent='완료되지 않은 실행은 실험 기록에 저장하지 않습니다.';notify('학습을 중지했습니다.');
}
function renderRuns(){
  $('results-empty').hidden=runs.length>0;$('results-content').hidden=!runs.length;$('export-results').disabled=!runs.length;
  if(!runs.length)return;
  if(!runs.some(r=>r.id===selectedRunId))selectedRunId=runs[0].id;
  $('run-count').textContent=runs.length+' RUN'+(runs.length>1?'S':'');
  const tbody=$('runs-table');tbody.replaceChildren();
  for(const run of runs){
    const tr=document.createElement('tr');tr.classList.toggle('selected',run.id===selectedRunId);
    const first=document.createElement('td');const button=document.createElement('button');button.type='button';button.textContent=run.name;
    button.setAttribute('aria-pressed',String(run.id===selectedRunId));const date=document.createElement('small');date.textContent=new Date(run.createdAt).toLocaleString('ko-KR',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'});
    button.append(date);button.addEventListener('click',()=>{selectedRunId=run.id;renderRuns();});first.append(button);tr.append(first);
    const vals=['×'+run.options.factor,run.options.epochs,pct(run.metrics.validation.f1),pct(run.metrics.test.f1),pct(run.metrics.test.iou),run.fingerprint,(run.elapsedMs/1000).toFixed(2)+'초'];
    vals.forEach((value,i)=>{const td=document.createElement('td');if(i===5){const code=document.createElement('code');code.textContent=value;td.append(code);}else td.textContent=value;tr.append(td);});
    tbody.append(tr);
  }
  const differing=new Set(runs.map(r=>r.fingerprint)).size>1;
  $('comparison-note').classList.toggle('warning',differing);
  $('comparison-note').textContent=(differing?'라벨 버전이 다른 실행이 포함되어 있습니다. 학습 조건의 차이를 고려해 비교하세요. ':'최신 실행 순서입니다. 모델 선택에는 검증 F1을 먼저 참고하고 같은 라벨 버전·학습 횟수를 비교하세요. ')+'같은 합성 평가군을 반복 열람하므로 최종 출시용 독립 평가를 대신하지 않습니다.';
  const selected=runs.find(r=>r.id===selectedRunId);$('selected-run-title').textContent=selected.name;
  const metrics=$('results-metrics');metrics.replaceChildren();
  for(const [label,key,description] of [['Test F1','f1','정밀도·재현율의 조화 평균'],['Test IoU','iou','정답·예측 영역의 겹침'],['정밀도','precision','예측 양성 중 정답 비율'],['재현율','recall','실제 양성 중 탐지 비율']]){
    const card=document.createElement('article');card.className='stat-card';const heading=document.createElement('span');heading.textContent=label;const value=document.createElement('strong');value.textContent=pct(selected.metrics.test[key]);const foot=document.createElement('p');foot.textContent=description;card.append(heading,value,foot);metrics.append(card);
  }
  renderPrediction();
}
function renderPrediction(){
  const run=runs.find(r=>r.id===selectedRunId);if(!run)return;
  const tile=dataset.tiles.find(t=>t.id===$('prediction-tile').value)||dataset.tiles.find(t=>t.split==='test');
  const input=degradeImage(tile.rgb,SIZE,SIZE,run.options.factor);
  const probs=predictMask(run.model,input,SIZE,SIZE);
  const predicted=Uint8Array.from(probs,p=>p>=.5?1:0);
  drawRGB($('prediction-input'),input);drawRGB($('prediction-label'),input,SIZE,SIZE,tile.mask);drawRGB($('prediction-output'),input,SIZE,SIZE,predicted,[72,204,238],.6);
}
function openGuide(){if(!$('guide-dialog').open)$('guide-dialog').showModal();}

initializeOfficialView();
loadState();
setupTileOptions();renderHero();renderStats();renderReview();renderResolution();renderFingerprint();renderRuns();route();
window.addEventListener('hashchange',route);
document.querySelector('.skip-link').addEventListener('click',event=>{event.preventDefault();$('main').focus();$('main').scrollIntoView({block:'start'});});
$$('[data-guide]').forEach(button=>button.addEventListener('click',openGuide));
$('close-guide').addEventListener('click',()=>$('guide-dialog').close());
$('guide-start').addEventListener('click',()=>{$('guide-dialog').close();location.hash='experiments';});
$('guide-dialog').addEventListener('click',event=>{const box=$('guide-dialog').getBoundingClientRect();if(event.target===$('guide-dialog')&&(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom))$('guide-dialog').close();});
$('export-dataset').addEventListener('click',exportLabels);$('export-labels').addEventListener('click',exportLabels);$('export-results').addEventListener('click',exportResults);
$$('[data-split]').forEach(button=>button.addEventListener('click',()=>{splitFilter=button.dataset.split;$$('[data-split]').forEach(b=>{const selected=b===button;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',String(selected));});renderReview();}));
$$('[data-brush]').forEach(button=>button.addEventListener('click',()=>{brushValue=Number(button.dataset.brush);$$('[data-brush]').forEach(b=>{const selected=b===button;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',String(selected));});}));
$('show-overlay').addEventListener('change',drawEditor);
$('brush-size').addEventListener('input',()=>{brushSize=Number($('brush-size').value);$('brush-size-value').textContent=brushSize+' px';drawEditor();});
$('save-labels').addEventListener('click',()=>saveLabels());
$('reset-mask').addEventListener('click',()=>{if(!canEdit())return;currentTile().mask=new Uint8Array(originalMasks.get(currentTileId));markDirty();drawEditor();notify('이 타일의 합성 원본 라벨로 되돌렸습니다. 저장하면 반영됩니다.');});
const editor=$('editor-canvas');
editor.addEventListener('pointerdown',event=>{if(!canEdit()||event.button!==0)return;event.preventDefault();painting=true;editor.setPointerCapture(event.pointerId);editor.focus({preventScroll:true});cursor={...pointerPosition(event),visible:true};lastPoint={...cursor};stamp(cursor.x,cursor.y);drawEditor();});
editor.addEventListener('pointermove',event=>{cursor={...pointerPosition(event),visible:true};if(painting&&canEdit()){paintLine(lastPoint||cursor,cursor);lastPoint={...cursor};}drawEditor();$('canvas-coordinate').textContent='x '+cursor.x+' · y '+cursor.y+' / 64 × 64';});
for(const type of ['pointerup','pointercancel'])editor.addEventListener(type,()=>{painting=false;lastPoint=null;});
editor.addEventListener('pointerleave',()=>{if(!painting){cursor.visible=false;drawEditor();}$('canvas-coordinate').textContent='64 × 64 px';});
editor.addEventListener('focus',()=>{cursor.visible=true;drawEditor();});editor.addEventListener('blur',()=>{cursor.visible=false;drawEditor();});
editor.addEventListener('keydown',event=>{if(!canEdit())return;const keys={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};if(keys[event.key]){event.preventDefault();cursor.x=Math.max(0,Math.min(SIZE-1,cursor.x+keys[event.key][0]));cursor.y=Math.max(0,Math.min(SIZE-1,cursor.y+keys[event.key][1]));cursor.visible=true;drawEditor();}else if(event.code==='Space'){event.preventDefault();stamp(cursor.x,cursor.y);drawEditor();}});
$$('[data-factor]').forEach(button=>button.addEventListener('click',()=>{if(activeRun)return;factor=Number(button.dataset.factor);$$('[data-factor]').forEach(b=>{const selected=b===button;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',String(selected));});renderResolution();}));
$('experiment-tile').addEventListener('change',renderResolution);$('prediction-tile').addEventListener('change',renderPrediction);
$('epochs').addEventListener('input',()=>$('epochs-value').textContent=$('epochs').value+' epoch');
$('training-form').addEventListener('submit',startTraining);$('stop-training').addEventListener('click',stopTraining);
if(pendingMessages.length)notify(pendingMessages.join(' '));
