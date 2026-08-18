import { currentTaskId, readApiResponse } from './segmentation-api';

const initializeSemanticLayerSession = async () => {
    const taskId = currentTaskId();
    if (!taskId) return;

    const endpoint = `/api/tasks/${encodeURIComponent(taskId)}/layers/cleanup`;
    const response = await fetch(endpoint, { method: 'POST' });
    await readApiResponse(response);

    window.addEventListener('pagehide', () => {
        navigator.sendBeacon(endpoint);
    }, { once: true });
};

export { initializeSemanticLayerSession };
