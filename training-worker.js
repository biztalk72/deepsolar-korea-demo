import {trainModel} from './ml-core.js';

self.onmessage = async ({data}) => {
  if (!data || data.type !== 'train') return;
  const {runId, tiles, options} = data;
  try {
    const result = await trainModel(tiles, options, progress => self.postMessage({type: 'progress', runId, ...progress}));
    self.postMessage({type: 'complete', runId, result});
  } catch (error) {
    self.postMessage({type: 'error', runId, message: error instanceof Error ? error.message : '브라우저 학습에 실패했습니다.'});
  }
};
