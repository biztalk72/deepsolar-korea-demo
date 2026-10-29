/** Read-only presentation of recorded CPU training on official real imagery samples. */
const byId = id => document.getElementById(id);
const FORMAT = 'deepsolar-official-sample-results-v1';
let report = null;
let pending = null;
let initialized = false;
const ratioKeys = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1'];
const countKeys = ['tp', 'fp', 'fn', 'tn'];
const finite = value => typeof value === 'number' && Number.isFinite(value);
const percent = value => finite(value) ? (value * 100).toFixed(1) + '%' : '—';
const text = value => value === undefined || value === null || value === '' ? '—' : String(value);
const setText = (id, value) => { byId(id).textContent = text(value); };
function element(tag, value, className) {
  const node = document.createElement(tag);
  if (value !== undefined) node.textContent = text(value);
  if (className) node.className = className;
  return node;
}
function definition(target, label, value, isCode = false) {
  target.append(element('dt', label));
  const dd = element('dd');
  dd.append(element(isCode ? 'code' : 'span', value));
  target.append(dd);
}
function computedMetrics(metric) {
  if (!metric || !countKeys.every(k => Number.isInteger(metric[k]) && metric[k] >= 0)) throw new Error('분류 오류 집계 형식이 올바르지 않습니다.');
  const {tp, fp, fn, tn} = metric;
  const n = tp + fp + fn + tn;
  if (!n || !Number.isInteger(metric.n) || metric.n !== n) throw new Error('영상 개수와 분류 오류 집계가 일치하지 않습니다.');
  const actual = {n,tp,fp,fn,tn,
    accuracy:(tp+tn)/n,
    balanced_accuracy:((tp+fn ? tp/(tp+fn) : 0)+(tn+fp ? tn/(tn+fp) : 0))/2,
    precision:tp+fp ? tp/(tp+fp) : 0,
    recall:tp+fn ? tp/(tp+fn) : 0,
    f1:2*tp+fp+fn ? 2*tp/(2*tp+fp+fn) : 0};
  for (const key of ratioKeys) {
    if (metric[key] !== undefined && (!finite(metric[key]) || Math.abs(metric[key]-actual[key]) > 1e-6)) throw new Error('분류 지표와 오류 집계가 일치하지 않습니다.');
  }
  if (metric.accuracy_ci95 !== undefined && (!Array.isArray(metric.accuracy_ci95) || metric.accuracy_ci95.length !== 2 || !metric.accuracy_ci95.every(v => finite(v) && v>=0 && v<=1) || metric.accuracy_ci95[0] > metric.accuracy_ci95[1])) throw new Error('정확도 구간 형식이 올바르지 않습니다.');
  return {...metric,...actual};
}
export function validateOfficialReport(data) {
  if (!data || data.schema !== FORMAT || data.status !== 'completed' || !data.dataset || !data.model) throw new Error('완료된 공식 샘플 학습 기록이 아닙니다.');
  const result = {...data,metrics:{}};
  for (const split of ['train','validation','test']) result.metrics[split] = computedMetrics(data.metrics?.[split]);
  const splits = data.dataset.split_counts;
  for (const split of ['train','validation','test']) {
    const count = splits?.[split];
    if (!count || !['total','positive','negative'].every(k => Number.isInteger(count[k]) && count[k]>=0) || count.total !== count.positive + count.negative || count.total !== result.metrics[split].n) throw new Error('분할 영상 개수가 평가 기록과 일치하지 않습니다.');
  }
  if (!Array.isArray(data.resolution_stress) || !data.resolution_stress.length) throw new Error('입력 해상도 시험 기록이 없습니다.');
  const factors = new Set();
  result.resolution_stress = data.resolution_stress.map(row => {
    if (!row || ![1,2,4,8].includes(row.factor) || factors.has(row.factor)) throw new Error('해상도 배율 기록이 올바르지 않습니다.');
    factors.add(row.factor);
    const metrics = computedMetrics(row.metrics);
    if (metrics.n !== result.metrics.test.n) throw new Error('해상도별 시험 영상 수가 서로 다릅니다.');
    if (row.factor === 1 && !countKeys.every(k => metrics[k] === result.metrics.test[k])) throw new Error('원본 입력의 시험 지표가 일치하지 않습니다.');
    return {...row,metrics};
  }).sort((a,b)=>a.factor-b.factor);
  if (!factors.has(1)) throw new Error('원본 입력 평가가 없습니다.');
  result.base_predictions = (Array.isArray(data.predictions) ? data.predictions : []).filter(row => row.split === 'test' && (row.factor === undefined || row.factor === 1));
  if (result.base_predictions.length !== result.metrics.test.n || !result.base_predictions.every(row => typeof row.sample_id === 'string' && [0,1].includes(row.label) && [0,1].includes(row.prediction) && finite(row.probability) && row.probability>=0 && row.probability<=1)) throw new Error('시험 영상별 예측 기록이 올바르지 않습니다.');
  const counted = {tp:0,fp:0,fn:0,tn:0};
  for (const row of result.base_predictions) counted[row.prediction ? (row.label ? 'tp':'fp') : (row.label ? 'fn':'tn')]++;
  if (!countKeys.every(k => counted[k] === result.metrics.test[k])) throw new Error('영상별 예측과 오류 집계가 일치하지 않습니다.');
  if (!finite(data.model.threshold) || data.model.threshold<0 || data.model.threshold>1) throw new Error('분류 임계값 기록이 올바르지 않습니다.');
  return result;
}
function wilson(correct, total) {
  const z = 1.959963984540054, p = correct/total, denominator = 1+z*z/total;
  const center = (p+z*z/(2*total))/denominator;
  const margin = z*Math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator;
  return [Math.max(0,center-margin),Math.min(1,center+margin)];
}
function safeLink(url, label) {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:') return null;
    const link = element('a', label, 'button secondary small');
    link.href = parsed.href; link.target = '_blank'; link.rel = 'noopener noreferrer';
    return link;
  } catch { return null; }
}
function renderReport(data) {
  const ds = data.dataset, model = data.model, metric = data.metrics.test;
  const total = Object.values(ds.split_counts).reduce((sum,row)=>sum+row.total,0);
  setText('official-dataset-title', ds.title || 'DeepSolar++ 공식 공개 예제');
  const inventoryNote = Number.isInteger(ds.total_available_images) && ds.total_available_images > total ? ' 공개 묶음 '+ds.total_available_images+'개 중 이번 실험의 학습·평가 대상은 '+total+'개입니다.' : '';
  setText('official-dataset-note', (ds.note || '공개 예제의 영상 단위 태양광 유무 라벨로 학습하고, 공식 분할에 따라 평가한 기록입니다.') + inventoryNote);
  setText('official-total-count', total+'장');
  const date = new Date(data.generated_at);
  setText('official-study-time', Number.isFinite(date.getTime()) ? new Intl.DateTimeFormat('ko-KR',{timeZone:'Asia/Seoul',dateStyle:'medium',timeStyle:'short'}).format(date)+' KST' : '실행 일시 미기록');
  setText('official-status', '실제 CPU 학습 완료');
  setText('official-test-n', '시험 '+metric.n+'장');
  setText('official-sample-caution', '시험 영상은 '+metric.n+'장입니다. 한 장의 분류 결과만 바뀌어도 정확도가 '+(100/metric.n).toFixed(1)+'%p 달라집니다. 원본 DeepSolar 논문 재현이나 한국 지역의 성능 평가를 의미하지 않습니다.');
  setText('official-overview-summary', '실제 공개 영상 '+total+'장 · 학습 '+ds.split_counts.train.total+' / 검증 '+ds.split_counts.validation.total+' / 시험 '+ds.split_counts.test.total+' · 소량 예제 실행 기록');
  const cards = byId('official-metrics'); cards.replaceChildren();
  for (const [label,key,detail] of [['시험 F1','f1','영상 단위 정밀도·재현율의 조화 평균'],['정밀도','precision','태양광 있음 예측 중 실제 양성 비율'],['재현율','recall','실제 양성 영상 중 찾아낸 비율'],['균형 정확도','balanced_accuracy','양성·음성 재현율의 평균']]) {
    const card = element('article',undefined,'stat-card');
    card.append(element('span',label),element('strong',percent(metric[key])),element('p',detail));cards.append(card);
  }
  const confusion = byId('official-confusion'); confusion.replaceChildren();
  for (const [label,key,style] of [['양성 정답 · TP','tp','correct'],['놓친 양성 · FN','fn','incorrect'],['잘못된 양성 · FP','fp','incorrect'],['음성 정답 · TN','tn','correct']]) {
    const card = element('div',undefined,'confusion-count '+style);
    card.append(element('span',label),element('strong',metric[key]+'장'));confusion.append(card);
  }
  const interval = metric.accuracy_ci95 || wilson(metric.tp+metric.tn,metric.n);
  const confidence = byId('official-confidence');confidence.replaceChildren();
  confidence.append(element('strong','시험 정확도 '+percent(metric.accuracy)+' · '+(metric.tp+metric.tn)+'/'+metric.n+'장 정답'));
  confidence.append(element('span','Wilson 95% 참고 구간 '+percent(interval[0])+'–'+percent(interval[1])));
  confidence.append(element('small','영상이 서로 독립이라는 가정의 참고 구간입니다. 촬영 위치의 독립성은 확인되지 않았으며 한국 일반화 성능의 범위가 아닙니다.'));
  const splitTable = byId('official-splits');splitTable.replaceChildren();
  for (const [key,label] of [['train','학습'],['validation','검증'],['test','시험']]) {
    const counts = ds.split_counts[key], tr = element('tr');
    for (const value of [label,counts.total+'장',counts.positive+'장',counts.negative+'장']) tr.append(element('td',value));
    splitTable.append(tr);
  }
  const details = byId('official-model-details');details.replaceChildren();
  definition(details,'모델',model.name);
  definition(details,'사전학습',model.pretraining);
  definition(details,'학습한 부분',model.trainable_component || '분류 헤드');
  if (Number.isInteger(model.trainable_parameters)) definition(details,'학습 파라미터',model.trainable_parameters.toLocaleString('ko-KR')+'개');
  if (model.backbone_finetuned === false) definition(details,'특징 추출기', '고정 · 추가 학습 없음');
  const size = Array.isArray(model.input_size) ? model.input_size.join(' × ') : model.input_size;
  definition(details,'모델 입력 크기',size === undefined ? undefined : size+' px');
  definition(details,'학습 횟수',model.epochs === undefined ? undefined : model.epochs+' epoch');
  definition(details,'선택 규칙',model.selection_rule);
  definition(details,'고정 임계값',model.threshold);
  definition(details,'난수 시드',model.seed);
  if (finite(model.elapsed_seconds)) definition(details,'전체 실행 시간',model.elapsed_seconds.toFixed(2)+'초');
  const stress = byId('official-resolution-rows');stress.replaceChildren();
  for (const row of data.resolution_stress) {
    const m = row.metrics,tr=element('tr');
    tr.append(element('td',row.factor===1 ? '원본 ×1' : row.factor+'배 축소'));
    const f1Cell = element('td'); const bar = element('span',undefined,'official-score-bar');bar.setAttribute('aria-hidden','true');
    const fill = element('i');fill.style.width=(m.f1*100)+'%';bar.append(fill);f1Cell.append(element('strong',percent(m.f1)),bar);tr.append(f1Cell);
    for(const value of [percent(m.precision),percent(m.recall),percent(m.balanced_accuracy),(m.tp+m.tn)+' / '+m.n])tr.append(element('td',value));
    stress.append(tr);
  }
  setText('official-resolution-note', data.resolution_note || '원본 영상의 면적 평균 축소 후 확대를 적용한 입력 열화 시험입니다. 실제 센서·GSD나 한국 지역의 저해상도 성능을 재현하지 않습니다.');
  const predictionRows=byId('official-prediction-rows');predictionRows.replaceChildren();
  for (const row of data.base_predictions) {
    const tr=element('tr');tr.append(element('td',row.sample_id));
    tr.append(element('td',row.label===1?'태양광 있음':'없음'));
    tr.append(element('td',row.probability.toFixed(4)));
    tr.append(element('td',row.prediction===1?'태양광 있음':'없음'));
    const match=element('td');match.append(element('span',row.label===row.prediction?'일치':'오류','badge '+(row.label===row.prediction?'muted':'official-error-badge')));tr.append(match);predictionRows.append(tr);
  }
  setText('official-rights-note', ds.rights_note || '원본 사진은 이 사이트에 재게시하지 않으며, 출처와 계산 결과를 제공합니다.');
  const source=byId('official-source-details');source.replaceChildren();
  definition(source,'자료',ds.title);
  definition(source,'원본 코드 커밋',ds.repository_commit,true);
  definition(source,'다운로드 SHA-256',ds.archive_sha256,true);
  if (ds.manifest_sha256) definition(source,'학습 manifest SHA-256',ds.manifest_sha256,true);
  if (ds.dataset_fingerprint) definition(source,'데이터 SHA-256',ds.dataset_fingerprint,true);
  if (model.preprocessing) definition(source,'입력 전처리',model.preprocessing);
  if (model.encoder_weights_sha256) definition(source,'기초 가중치 SHA-256',model.encoder_weights_sha256,true);
  definition(source,'데이터 분리 검토',ds.leakage_note);
  if (ds.label_note) definition(source,'정답 형태',ds.label_note);
  const links=byId('official-source-links');links.replaceChildren();
  for (const [url,label] of [[ds.source_url,'공식 코드·자료 안내 ↗'],[ds.download_url,'공식 예제 다운로드 ↗']]) {const link=safeLink(url,label);if(link)links.append(link);}
  const limits=byId('official-limitations');limits.replaceChildren();
  const caveats = Array.isArray(data.limitations) && data.limitations.length ? data.limitations : [
    '소량의 공식 공개 예제에서 실행을 검증한 결과입니다.',
    '2018년 원본 DeepSolar 전체 학습·시험 데이터의 재현 결과가 아닙니다.',
    '이미지 단위 분류이며 픽셀 마스크 정확도, 패널 개수나 설비 용량을 측정하지 않습니다.',
    '한국 영상의 학습·평가와 상업 적용 검증은 별도 데이터로 수행해야 합니다.'
  ];
  for (const item of caveats) if (typeof item==='string') limits.append(element('li',item));
  renderKoreaStatus(data.korea_data);
}
function renderKoreaStatus(data) {
  const section=byId('official-korea-section');
  if(!section)return;
  section.hidden=!data;
  if(!data)return;
  setText('official-korea-title',data.title || '한국 영상 데이터 준비');
  setText('official-korea-summary',data.summary);
  const target=byId('official-korea-details');target.replaceChildren();
  definition(target,'준비 상태',data.status);
  if(Number.isInteger(data.image_count))definition(target,'확보 영상',data.image_count+'개');
  definition(target,'정답 라벨',data.label_status);
  definition(target,'학습 가능 상태',data.training_ready===true?'라벨·분할 검토 완료':'추가 라벨·분할 검토 필요');
  definition(target,'학습 결과',data.trained===true ? '별도 한국 실험 기록 참고' : '한국 영상 학습 결과 미포함');
  if(data.source_url){const link=safeLink(data.source_url,'한국 영상 자료 출처 ↗');if(link)target.append(link);}
}
export function updateOfficialScope(route) {
  const official=route==='official',overview=route==='overview';
  document.body.dataset.study=official?'official':overview?'mixed':'synthetic';
  setText('scope-title',official?'DeepSolar++ 공식 예제 · 로컬 학습 결과':overview?'공식 예제 결과 · 합성 브라우저 실험':'GitHub Pages · 합성 데이터 실험');
  setText('scope-description',official?'소량의 실제 위성·항공 영상 분류 기록입니다. 한국 영상의 성능과 구분해 보세요.':overview?'공식 샘플의 기록과 합성 영상의 직접 학습을 각각 확인할 수 있습니다.':'합성 영상의 학습은 현재 브라우저에서 실행됩니다. 실제 위성 탐지 정확도와 구분해 보세요.');
  setText('environment-mode',official?'로컬 CPU 결과':overview?'실험 · 결과 검토':'브라우저 연산');
  setText('environment-description',official?'공식 공개 예제의 완료된 학습·평가 기록':overview?'실제 공개 예제와 합성 데이터 실험실':'합성 데이터로 검증하는 경량 모델 실험실');
  setText('environment-data',official?'공식 위성·항공 예제':overview?'공식 예제 · 합성':'합성 영상');
  setText('environment-model',official?'영상 분류 모델':overview?'로컬 · 브라우저':'로지스틱 기준선');
}
export async function loadOfficialResults(force=false) {
  if(report&&!force)return report;
  if(pending)return pending;
  byId('official-loading').hidden=false;byId('official-error').hidden=true;
  byId('official-content').hidden=true;byId('official-download').disabled=true;
  pending=(async()=>{
    const controller=new AbortController();
    const timer=setTimeout(()=>controller.abort(),12000);
    try {
      const response=await fetch(new URL('./real-sample-results.json',import.meta.url),{cache:'no-store',signal:controller.signal});
      if(!response.ok)throw new Error(response.status===404?'완료된 학습 기록 파일이 아직 게시되지 않았습니다.':'학습 기록을 받을 수 없습니다. HTTP '+response.status);
      const data=validateOfficialReport(await response.json());
      renderReport(data);report=data;
      byId('official-content').hidden=false;byId('official-download').disabled=false;
      return data;
    } catch(error) {
      byId('official-error').hidden=false;
      setText('official-error-message',error.name==='AbortError'?'학습 기록 요청 시간이 초과되었습니다. 다시 불러오기를 눌러주세요.':error.message);
      setText('official-overview-summary','학습 기록을 확인할 수 없습니다. 공식 샘플 결과 화면에서 다시 불러올 수 있습니다.');
      return null;
    } finally {clearTimeout(timer);byId('official-loading').hidden=true;pending=null;}
  })();
  return pending;
}
export function initializeOfficialView() {
  if(initialized)return;initialized=true;
  byId('official-retry').addEventListener('click',()=>loadOfficialResults(true));
  byId('official-download').addEventListener('click',()=>{
    if(!report)return;
    const link=document.createElement('a');
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json;charset=utf-8'}));
    link.href=url;link.download='deepsolar-official-sample-results.json';document.body.append(link);link.click();link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),1500);
  });
  loadOfficialResults();
}
