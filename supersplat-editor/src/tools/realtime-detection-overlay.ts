import { Events } from '../events';
import { Scene } from '../scene';

type Detection = { category: string; score: number; bbox: [number, number, number, number] };
type TrackedDetection = Detection & { missed: number };

const REQUEST_INTERVAL_MS = 80;
const MOVEMENT_GRACE_MS = 250;
const CURRENT_BOX_WEIGHT = 0.9;

const LABELS: Record<string, string> = {
    bottle: '瓶子',
    cup: '杯子',
    chair: '椅子',
    bed: '床',
    couch: '沙发',
    tv: '电视',
    laptop: '笔记本电脑',
    keyboard: '键盘',
    mouse: '鼠标',
    'cell phone': '手机',
    book: '书',
    vase: '花瓶',
    clock: '时钟',
    refrigerator: '冰箱',
    microwave: '微波炉',
    oven: '烤箱',
    sink: '水槽',
    toilet: '马桶',
    'dining table': '餐桌',
    pottedplant: '盆栽'
};

class RealtimeDetectionOverlay {
    private root: HTMLDivElement;
    private boxes: HTMLDivElement;
    private toggle: HTMLButtonElement;
    private lastMatrix: Float32Array | null = null;
    private lastMovement = 0;
    private lastRequest = 0;
    private requestActive = false;
    private frameId = 0;
    private displayedFrame = 0;
    private paused = false;
    private unavailable = false;
    private tracked: TrackedDetection[] = [];
    private enabled = localStorage.getItem('realtime-detection-enabled') !== 'false';
    private pointerActive = false;
    private lastRendered = 0;

    constructor(private events: Events, private scene: Scene, host: HTMLElement) {
        this.root = document.createElement('div');
        this.root.className = 'realtime-detection-overlay';
        Object.assign(this.root.style, {
            position: 'absolute', inset: '0', pointerEvents: 'none', zIndex: '8', overflow: 'hidden'
        });
        this.boxes = document.createElement('div');
        this.boxes.className = 'realtime-detection-boxes';
        Object.assign(this.boxes.style, { position: 'absolute', inset: '0' });
        this.toggle = document.createElement('button');
        this.toggle.className = 'realtime-detection-toggle';
        this.toggle.type = 'button';
        Object.assign(this.toggle.style, {
            position: 'absolute',
            left: '12px',
            top: '12px',
            zIndex: '2',
            pointerEvents: 'auto',
            padding: '5px 9px',
            border: '1px solid #4f6670',
            borderRadius: '4px',
            color: '#fff',
            background: 'rgba(24, 32, 38, .86)',
            fontSize: '12px',
            cursor: 'pointer'
        });
        this.toggle.addEventListener('pointerdown', (event) => {
            event.preventDefault();
            event.stopImmediatePropagation();
            this.enabled = !this.enabled;
            localStorage.setItem('realtime-detection-enabled', String(this.enabled));
            if (!this.enabled) this.clear();
            this.updateStatus();
        }, true);
        this.root.append(this.boxes, this.toggle);
        this.updateStatus();
        host.appendChild(this.root);
        window.setInterval(() => this.update(), 100);
        events.on('camera.moved', () => this.markMovement());
        window.addEventListener('pointerdown', (event) => {
            if (this.root.contains(event.target as Node)) return;
            this.pointerActive = true;
            this.markMovement();
        }, true);
        window.addEventListener('pointermove', () => {
            if (this.pointerActive) this.markMovement();
        }, true);
        window.addEventListener('pointerup', () => {
            this.pointerActive = false;
            this.markMovement();
        }, true);
        window.addEventListener('wheel', (event) => {
            if (!this.root.contains(event.target as Node)) this.markMovement();
        }, { capture: true, passive: true });
        events.on('update', () => this.update());
        events.on('tool.activated', (name: string | null) => {
            this.paused = name === 'segmentation' || name === 'sceneUnderstanding';
            if (this.paused) this.clear();
        });
        events.on('tool.deactivated', (name: string) => {
            if (name === 'segmentation' || name === 'sceneUnderstanding') {
                this.paused = false;
                this.lastMatrix = null;
            }
        });
    }

    private update() {
        if (!this.enabled || this.paused || this.unavailable || document.hidden) return;
        const matrix = this.scene.camera.worldTransform.data;
        const current = new Float32Array(matrix);
        const moved = this.lastMatrix !== null && current.some(
            (value, index) => Math.abs(value - this.lastMatrix[index]) > 1e-4
        );
        this.lastMatrix = current;
        const now = performance.now();
        if (moved) this.lastMovement = now;
        const moving = moved || (now - this.lastMovement < MOVEMENT_GRACE_MS);
        if (!moving) {
            if (now - Math.max(this.lastMovement, this.lastRendered) > 1000 && this.tracked.length) {
                this.clear();
            }
            return;
        }
        if (!this.requestActive && now - this.lastRequest >= REQUEST_INTERVAL_MS) void this.detect();
    }

    private markMovement() {
        if (this.paused || !this.enabled || this.unavailable) return;
        const now = performance.now();
        this.lastMovement = now;
        if (!this.requestActive && now - this.lastRequest >= REQUEST_INTERVAL_MS) void this.detect();
    }

    private async detect() {
        this.requestActive = true;
        this.updateStatus('检测中');
        this.lastRequest = performance.now();
        const frameId = ++this.frameId;
        try {
            const rgba: Uint8Array = await this.events.invoke('render.offscreen', 640, 360);
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 360;
            canvas.getContext('2d')?.putImageData(
                new ImageData(new Uint8ClampedArray(rgba), 640, 360), 0, 0
            );
            const blob = await new Promise<Blob>((resolve, reject) => {
                canvas.toBlob(value => (value ? resolve(value) : reject(new Error('截图编码失败'))), 'image/jpeg', 0.75);
            });
            const form = new FormData();
            form.append('frame_id', String(frameId));
            form.append('image', blob, 'camera.jpg');
            const response = await fetch('/api/realtime/detect', { method: 'POST', body: form });
            if (response.status === 409) return;
            const body = await response.json().catch(() => ({}));
            if (response.status === 503) {
                this.unavailable = true;
                this.clear();
                this.updateStatus('不可用');
                return;
            }
            if (!response.ok) throw new Error(body.detail ?? `实时检测请求失败 ${response.status}`);
            if (body.frame_id < this.displayedFrame) return;
            this.displayedFrame = body.frame_id;
            this.render(this.smooth(body.detections ?? []));
        } catch (error) {
            this.clear();
            this.updateStatus('错误');
            this.toggle.title = error instanceof Error ? error.message : '实时检测请求失败';
        } finally {
            this.requestActive = false;
            if (!this.toggle.textContent?.includes('错误')) this.updateStatus();
        }
    }

    private smooth(detections: Detection[]) {
        const unused = new Set(this.tracked.map((_, index) => index));
        const next = detections.map((detection): TrackedDetection => {
            let best = -1;
            let bestIou = 0.3;
            for (const index of unused) {
                const previous = this.tracked[index];
                if (previous.category !== detection.category) continue;
                const iou = this.boxIou(previous.bbox, detection.bbox);
                if (iou > bestIou) {
                    best = index;
                    bestIou = iou;
                }
            }
            if (best < 0) return { ...detection, missed: 0 };
            unused.delete(best);
            const previous = this.tracked[best];
            return {
                ...detection,
                bbox: detection.bbox.map(
                    (value, index) => value * CURRENT_BOX_WEIGHT +
                        previous.bbox[index] * (1 - CURRENT_BOX_WEIGHT)
                ) as Detection['bbox'],
                missed: 0
            };
        });
        for (const index of unused) {
            const previous = this.tracked[index];
            if (previous.missed < 1) next.push({ ...previous, missed: previous.missed + 1 });
        }
        this.tracked = next;
        return next;
    }

    private boxIou(a: Detection['bbox'], b: Detection['bbox']) {
        const left = Math.max(a[0], b[0]);
        const top = Math.max(a[1], b[1]);
        const right = Math.min(a[2], b[2]);
        const bottom = Math.min(a[3], b[3]);
        const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
        const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
        const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
        return intersection / Math.max(areaA + areaB - intersection, 1e-6);
    }

    private render(detections: Detection[]) {
        this.lastRendered = performance.now();
        this.boxes.replaceChildren(...detections.map((detection) => {
            const [x1, y1, x2, y2] = detection.bbox;
            const box = document.createElement('div');
            box.className = 'realtime-detection-box';
            Object.assign(box.style, {
                position: 'absolute',
                left: `${x1 * 100}%`,
                top: `${y1 * 100}%`,
                width: `${(x2 - x1) * 100}%`,
                height: `${(y2 - y1) * 100}%`,
                border: '2px solid #42d7ff',
                boxSizing: 'border-box'
            });
            const label = document.createElement('span');
            label.className = 'realtime-detection-label';
            label.textContent = `${LABELS[detection.category] ?? detection.category} ${(detection.score * 100).toFixed(0)}%`;
            Object.assign(label.style, {
                position: 'absolute',
                left: '-2px',
                top: '-24px',
                padding: '2px 5px',
                color: '#fff',
                background: 'rgba(0, 120, 170, .9)',
                fontSize: '12px',
                whiteSpace: 'nowrap'
            });
            box.appendChild(label);
            return box;
        }));
    }

    private clear() {
        this.tracked = [];
        this.boxes.replaceChildren();
    }

    private updateStatus(detail = '') {
        const state = this.unavailable ? '不可用' : this.enabled ? '开' : '关';
        this.toggle.textContent = `YOLO实时检测：${detail || state}`;
        this.toggle.title = '移动相机时显示预览框，最终结果以“分割”为准';
    }
}

export { RealtimeDetectionOverlay };
