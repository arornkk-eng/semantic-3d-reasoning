export const currentTaskId = () => new URLSearchParams(location.search).get('task_id');

export const readApiResponse = async <T = any>(response: Response): Promise<T> => {
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `请求失败 ${response.status}`);
    return body;
};
