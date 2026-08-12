import { FILTER_NEAREST, Mat4, Quat, Texture, Vec3 } from 'playcanvas';

import { SemanticIntersectState } from '../data-processor';
import { SH_C0, sigmoid } from '../color-grade';
import { ElementType } from '../element';
import { Events } from '../events';
import { Scene } from '../scene';
import { Splat } from '../splat';
import { State } from '../splat-state';
import { expandSemanticGaussianSeeds } from './semantic-gaussian-expansion';

type PromptPoint = { x: number; y: number; label: 0 | 1 };
type SemanticViewMask = { view_index: number; mask_url: string };
type SemanticInstance = {
    instance_id: string;
    category: string;
    category_zh: string;
    instance_index: number;
    score: number;
    mask_url: string;
    selected: boolean;
    depth_coverage?: number;
    view_support?: number;
    view_count?: number;
    view_masks?: SemanticViewMask[];
};

type CapturedView = {
    image: Blob;
    depth: Blob;
    depthValues: Float32Array;
    depthCoverage: Float32Array;
    metadata: {
        task_id: string | null;
        source_ply: string;
        viewport_width: number;
        viewport_height: number;
        capture_width: number;
        capture_height: number;
        near: number;
        far: number;
        projection: 'orthographic' | 'perspective';
        fov: number;
        camera_position: number[];
        camera_rotation: number[];
        view_matrix: number[];
        projection_matrix: number[];
    };
};

type ProjectedSplat = {
    splat: Splat;
    indices: Uint32Array;
    expansionCandidates: Uint32Array;
    expansionApplied: boolean;
};

class SegmentationTool {
    private events: Events;
    private scene: Scene;
    private host: HTMLElement;
    private root: HTMLDivElement;
    private overlay: HTMLImageElement;
    private markers: HTMLDivElement;
    private status: HTMLSpanElement;
    private nameInput: HTMLInputElement;
    private layerInfo: HTMLSpanElement;
    private sessionId: string | null = null;
    private points: PromptPoint[] = [];
    private label: 0 | 1 = 1;
    private busy = false;
    private active = false;
    private semanticResultId: string | null = null;
    private semanticInstances: SemanticInstance[] = [];
    private semanticViews: CapturedView[] = [];
    private promptView: CapturedView | null = null;
    private promptMaskUrl: string | null = null;
    private projectedInstances = new Map<string, ProjectedSplat[]>();

    constructor(events: Events, scene: Scene, host: HTMLElement) {
        this.events = events;
        this.scene = scene;
        this.host = host;
        this.root = document.createElement('div');
        this.root.id = 'segmentation-tool';
        this.root.innerHTML = `
            <img class="segmentation-mask prompt-mask" alt="">
            <div class="semantic-masks"></div>
            <div class="segmentation-markers"></div>
            <div class="segmentation-controls">
                <button data-action="auto" class="active">重新识别</button>
                <div class="semantic-list"></div>
                <input data-role="name" type="hidden" value="分割图层">
                <span data-role="layer-info">图层：待识别</span>
                <button data-action="project3d">投射到3D</button>
                <button data-action="expand3d">补全3D</button>
                <button data-action="refine3d">精细补全</button>
                <button data-action="separate3d">生成独立3D图层</button>
                <button data-action="confirm" class="primary">保存所选图层</button>
                <button data-action="cancel">取消</button>
                <span data-role="status">正在识别杯子、椅子、瓶子</span>
            </div>`;
        this.overlay = this.root.querySelector('.segmentation-mask');
        this.markers = this.root.querySelector('.segmentation-markers');
        this.status = this.root.querySelector('[data-role="status"]');
        this.nameInput = this.root.querySelector('[data-role="name"]');
        this.layerInfo = this.root.querySelector('[data-role="layer-info"]');
        document.addEventListener('pointerdown', this.onDocumentPointerDown, true);
        this.host.appendChild(this.root);
    }

    activate = () => {
        const taskId = new URLSearchParams(location.search).get('task_id');
        if (!taskId) {
            window.alert('当前模型缺少 task_id，无法保存分割图层');
            this.events.fire('tool.deactivate');
            return;
        }
        this.active = true;
        this.root.classList.add('active');
        void this.predictSemantic();
    };

    deactivate = () => {
        this.active = false;
        this.root.classList.remove('active');
        if (this.sessionId) void this.closeSession();
        this.reset();
    };

    private onRootClick = (event: Event) => {
        event.stopPropagation();
        const action = (event.target as HTMLElement).closest<HTMLButtonElement>('button')?.dataset.action;
        if (!action || this.busy) return;
        if (action === 'auto') {
            void this.predictSemantic();
        } else if (action === 'toggle') {
            const instance = this.semanticInstances.find(item => item.instance_id === (event.target as HTMLElement).closest<HTMLButtonElement>('button')?.dataset.instance);
            if (instance) {
                instance.selected = !instance.selected;
                this.renderSemanticResults();
            }
        } else if (action === 'positive' || action === 'negative') {
            this.label = action === 'positive' ? 1 : 0;
            this.root.querySelectorAll('[data-action="positive"], [data-action="negative"]').forEach((el) => {
                el.classList.toggle('active', (el as HTMLElement).dataset.action === action);
            });
        } else if (action === 'undo') {
            this.points.pop();
            this.renderMarkers();
            if (this.points.length) void this.predict();
            else this.clearMask();
        } else if (action === 'clear') {
            this.points = [];
            this.renderMarkers();
            this.clearMask();
        } else if (action === 'project3d') {
            void this.projectTo3D();
        } else if (action === 'expand3d') {
            void this.expand3DSelection();
        } else if (action === 'refine3d') {
            void this.refine3DSelection();
        } else if (action === 'separate3d') {
            void this.separate3DLayer();
        } else if (action === 'confirm') {
            void this.confirm();
        } else if (action === 'cancel') {
            this.events.fire('tool.deactivate');
        }
    };

    private onDocumentPointerDown = (event: PointerEvent) => {
        if (this.active && this.root.contains(event.target as Node)) {
            this.onRootClick(event);
        }
    };

    private onCanvasPointer = (event: PointerEvent) => {
        if (!this.active || this.busy || event.button !== 0) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        if (this.semanticResultId) {
            // A canvas prompt switches from automatic category masks to the
            // manual SAM session, so confirm must persist the prompt result.
            this.semanticResultId = null;
            this.semanticInstances = [];
            this.semanticViews = [];
            this.projectedInstances.clear();
            this.root.querySelector('.semantic-masks')?.replaceChildren();
            this.root.querySelector('.semantic-list')?.replaceChildren();
        }
        const rect = this.scene.canvas.getBoundingClientRect();
        this.points.push({
            x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
            y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
            label: this.label
        });
        this.renderMarkers();
        void this.predict();
    };

    private async ensureSession() {
        if (this.sessionId) return;
        const taskId = new URLSearchParams(location.search).get('task_id');
        const view = await this.captureView();
        this.promptView = view;
        const form = new FormData();
        form.append('image', view.image, 'camera-view.png');
        form.append('metadata', JSON.stringify(view.metadata));
        const response = await fetch('/api/segmentation/sessions', { method: 'POST', body: form });
        const body = await this.readResponse(response);
        this.sessionId = body.session_id;
        sessionStorage.setItem(`segmentation:${taskId}`, JSON.stringify({ sessionId: this.sessionId, points: this.points }));
    }

    private async predict() {
        this.setBusy(true, '分割计算中');
        try {
            await this.ensureSession();
            const response = await fetch(`/api/segmentation/sessions/${this.sessionId}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ points: this.points })
            });
            const body = await this.readResponse(response);
            if (this.sessionId) {
                this.projectedInstances.delete(this.sessionId);
            }
            this.promptMaskUrl = `${body.mask_url}?v=${Date.now()}`;
            this.overlay.src = this.promptMaskUrl;
            this.overlay.classList.add('visible');
            this.setStatus(`置信度 ${(body.score * 100).toFixed(1)}%`);
            const taskId = new URLSearchParams(location.search).get('task_id');
            sessionStorage.setItem(`segmentation:${taskId}`, JSON.stringify({ sessionId: this.sessionId, points: this.points }));
        } catch (error) {
            this.setStatus(error instanceof Error ? error.message : '分割失败');
        } finally {
            this.setBusy(false);
        }
    }

    private async confirm() {
        if (this.semanticResultId) {
            await this.confirmSemantic();
            return;
        }
        const sessionId = this.sessionId;
        if (!sessionId || !this.points.length) {
            this.setStatus('请先添加提示点');
            return;
        }
        this.setBusy(true, '保存图层中');
        try {
            const taskId = new URLSearchParams(location.search).get('task_id');
            const hasProjectedIndices = (this.projectedInstances.get(sessionId) ?? [])
            .some(projected => projected.indices.length > 0);
            if (!hasProjectedIndices) {
                if (!this.promptView || !this.promptMaskUrl) {
                    throw new Error('当前没有可投射的 mask');
                }
                const projectedCount = await this.projectInstancesTo3D([{
                    instance_id: sessionId,
                    view_masks: [{ view_index: 0, mask_url: this.promptMaskUrl }]
                }], [this.promptView]);
                if (!projectedCount) {
                    throw new Error('3D 投射结果为空，请修正 mask 或视角');
                }
            }
            const gaussianIndexSets = this.buildGaussianIndexSets([sessionId]);
            if (!gaussianIndexSets.some(item => item.indices.length > 0)) {
                throw new Error('没有可保存的 3D Gaussian 索引');
            }
            const response = await fetch(`/api/tasks/${taskId}/layers`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    name: this.nameInput.value.trim() || '分割图层',
                    gaussian_index_sets: gaussianIndexSets
                })
            });
            const layer = await this.readResponse(response);
            sessionStorage.removeItem(`segmentation:${taskId}`);
            this.sessionId = null;
            window.alert(`图层“${layer.name}”已保存`);
            this.events.fire('tool.deactivate');
        } catch (error) {
            this.setStatus(error instanceof Error ? error.message : '保存失败');
        } finally {
            this.setBusy(false);
        }
    }

    private async captureView(): Promise<CapturedView> {
        const params = new URLSearchParams(location.search);
        const image = await this.captureSceneBlob();
        const { width, height } = this.captureSize();
        const depthData: { depth: Float32Array; coverage: Float32Array } =
            await this.events.invoke('render.depthData', width, height);
        const depthValues = depthData.depth;
        const depthCopy = new Float32Array(depthValues.length);
        depthCopy.set(depthValues);
        const coverageCopy = new Float32Array(depthData.coverage.length);
        coverageCopy.set(depthData.coverage);
        const depth = new Blob([depthCopy.buffer], { type: 'application/octet-stream' });
        const camera = this.scene.camera.camera;
        const position = this.scene.camera.position;
        const rotation = this.scene.camera.mainCamera.getRotation();
        const captureSize = this.captureSize();
        return {
            image,
            depth,
            depthValues: depthCopy,
            depthCoverage: coverageCopy,
            metadata: {
                task_id: params.get('task_id'),
                source_ply: params.get('load') ?? '',
                viewport_width: this.scene.canvas.width,
                viewport_height: this.scene.canvas.height,
                capture_width: captureSize.width,
                capture_height: captureSize.height,
                near: this.scene.camera.near,
                far: this.scene.camera.far,
                projection: this.scene.camera.ortho ? 'orthographic' : 'perspective',
                fov: this.scene.camera.fov,
                camera_position: [position.x, position.y, position.z],
                camera_rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
                view_matrix: Array.from(camera.viewMatrix.data),
                projection_matrix: Array.from(camera.projectionMatrix.data)
            }
        };
    }

    private async captureSemanticViews() {
        const camera = this.scene.camera;
        const pose = this.events.invoke('camera.getPose');
        const basePosition = new Vec3(pose.position.x, pose.position.y, pose.position.z);
        const baseTarget = new Vec3(pose.target.x, pose.target.y, pose.target.z);
        const radial = new Vec3().sub2(basePosition, baseTarget);
        if (radial.length() <= 1e-8) {
            throw new Error('相机位置与轨道中心重合，无法捕获轨道视角');
        }
        const nearPole = Math.abs(new Vec3().copy(radial).normalize().dot(Vec3.UP)) > 0.98;
        const orbitAxis = nearPole ? Vec3.RIGHT : Vec3.UP;
        const lookUp = nearPole ? Vec3.BACK : Vec3.UP;
        const baseFov = camera.fov;
        const baseNear = camera.near;
        const baseFar = camera.far;
        const previousOverride = camera.poseOverride;
        const views: Awaited<ReturnType<SegmentationTool['captureView']>>[] = [];
        try {
            for (const angle of [0, -6, 6]) {
                const position = new Vec3();
                new Quat().setFromAxisAngle(orbitAxis, angle).transformVector(radial, position);
                position.add(baseTarget);
                const rotation = new Quat().setFromMat4(new Mat4().setLookAt(position, baseTarget, lookUp));
                camera.setPoseOverride({
                    position,
                    rotation,
                    fov: baseFov,
                    near: baseNear,
                    far: baseFar
                });
                await this.events.invoke('render.sortCurrentPose');
                views.push(await this.captureView());
            }
        } finally {
            camera.setPoseOverride(previousOverride);
            this.scene.forceRender = true;
        }
        return views;
    }

    private async captureSceneBlob(): Promise<Blob> {
        const { width, height } = this.captureSize();
        const rgba: Uint8Array = await this.events.invoke('render.offscreen', width, height);
        const pixels = new Uint8ClampedArray(rgba.length);
        pixels.set(rgba);
        const imageData = new ImageData(pixels, width, height);
        if (typeof OffscreenCanvas !== 'undefined') {
            const canvas = new OffscreenCanvas(width, height);
            const context = canvas.getContext('2d');
            if (!context) throw new Error('无法创建离屏画布');
            context.putImageData(imageData, 0, 0);
            return canvas.convertToBlob({ type: 'image/png' });
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d');
        if (!context) throw new Error('无法创建截图画布');
        context.putImageData(imageData, 0, 0);
        return new Promise<Blob>((resolve, reject) => {
            canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('视角截图失败')), 'image/png');
        });
    }

    private captureSize() {
        const sourceWidth = this.scene.canvas.width;
        const sourceHeight = this.scene.canvas.height;
        const scale = Math.min(1, 1280 / Math.max(sourceWidth, sourceHeight));
        return {
            width: Math.max(1, Math.round(sourceWidth * scale)),
            height: Math.max(1, Math.round(sourceHeight * scale))
        };
    }

    private async predictSemantic() {
        this.setBusy(true, '正在识别杯子、椅子、瓶子');
        try {
            const views = await this.captureSemanticViews();
            this.semanticViews = views;
            this.projectedInstances.clear();
            const metadata = {
                ...views[0].metadata,
                views: views.map(view => ({
                    near: view.metadata.near,
                    far: view.metadata.far,
                    projection: view.metadata.projection,
                    fov: view.metadata.fov,
                    camera_position: view.metadata.camera_position,
                    camera_rotation: view.metadata.camera_rotation,
                    view_matrix: view.metadata.view_matrix,
                    projection_matrix: view.metadata.projection_matrix
                }))
            };
            const form = new FormData();
            views.forEach((view, index) => {
                form.append('image', view.image, `camera-view-${index}.png`);
                form.append('depth', view.depth, `camera-depth-${index}.f32`);
            });
            form.append('metadata', JSON.stringify(metadata));
            const response = await fetch('/api/semantic/predict', { method: 'POST', body: form });
            const body = await this.readResponse(response);
            this.semanticResultId = body.result_id;
            this.semanticInstances = body.instances.map((item: SemanticInstance) => ({ ...item, selected: true }));
            this.renderSemanticResults();
            this.setStatus(this.semanticInstances.length ? `识别到 ${this.semanticInstances.length} 个物体` : '未识别到杯子、椅子或瓶子');
        } catch (error) {
            this.setStatus(error instanceof Error ? error.message : '自动识别失败');
        } finally {
            this.setBusy(false);
        }
    }

    private renderSemanticResults() {
        const selectedNames = this.semanticInstances
        .filter(item => item.selected)
        .map(item => `${item.category_zh}${item.instance_index}`);
        const layerName = selectedNames.join('+') || '未选择';
        this.nameInput.value = selectedNames.length ? layerName : '分割图层';
        this.layerInfo.textContent = `图层：${layerName}`;
        const masks = this.root.querySelector('.semantic-masks');
        masks?.replaceChildren(...this.semanticInstances.filter(item => item.selected).map(item => {
            const image = document.createElement('img');
            image.className = `segmentation-mask semantic-mask visible category-${item.category}`;
            image.src = `${item.mask_url}?v=${Date.now()}`;
            return image;
        }));
        const list = this.root.querySelector('.semantic-list');
        list?.replaceChildren(...this.semanticInstances.map(item => {
            const button = document.createElement('button');
            button.dataset.action = 'toggle';
            button.dataset.instance = item.instance_id;
            button.className = item.selected ? `selected category-${item.category}` : '';
            const support = item.view_count && item.view_count > 1 ? ` ${item.view_support}/${item.view_count}视角` : '';
            button.textContent = `${item.category_zh}${item.instance_index} ${(item.score * 100).toFixed(0)}%${support}`;
            return button;
        }));
    }

    private matrixFromArray(values: number[]) {
        if (values.length !== 16) throw new Error('相机矩阵必须包含 16 个数值');
        const matrix = new Mat4();
        matrix.data.set(values);
        return matrix;
    }

    private async loadMaskTexture(url: string, width: number, height: number) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`读取 mask 失败 ${response.status}`);
        const bitmap = await createImageBitmap(await response.blob());
        try {
            if (bitmap.width !== width || bitmap.height !== height) {
                throw new Error(`mask 尺寸 ${bitmap.width}x${bitmap.height} 与深度 ${width}x${height} 不一致`);
            }
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const context = canvas.getContext('2d');
            if (!context) throw new Error('无法创建 mask 画布');
            context.drawImage(bitmap, 0, 0);
            const texture = new Texture(this.scene.graphicsDevice, {
                name: 'semanticMask',
                mipmaps: false,
                minFilter: FILTER_NEAREST,
                magFilter: FILTER_NEAREST
            });
            texture.setSource(canvas);
            return texture;
        } finally {
            bitmap.close();
        }
    }

    private async projectInstancesTo3D(
        instances: Array<{ instance_id: string; view_masks: SemanticViewMask[] }>,
        views: CapturedView[]
    ) {
        const splats = (this.scene.getElementsByType(ElementType.splat) as Splat[]).filter(splat => splat.visible);
        if (!splats.length) throw new Error('当前场景没有可投射的 Gaussian 模型');

        const combined = new Map<Splat, Uint8Array>();
        splats.forEach(splat => combined.set(splat, new Uint8Array(splat.splatData.numSplats)));
        this.projectedInstances.clear();

        for (const instance of instances) {
            const projected: ProjectedSplat[] = [];
            for (const splat of splats) {
                const count = splat.splatData.numSplats;
                const positive = new Uint8Array(count);
                const visible = new Uint8Array(count);
                const maskHit = new Uint8Array(count);
                const visibleOutside = new Uint8Array(count);
                const frontMismatch = new Uint8Array(count);
                const behindMismatch = new Uint8Array(count);

                for (const viewMask of instance.view_masks) {
                    const view = views[viewMask.view_index];
                    if (!view) continue;
                    const { capture_width: width, capture_height: height } = view.metadata;
                    const texture = await this.loadMaskTexture(viewMask.mask_url, width, height);
                    let states: Uint8Array | null = null;
                    try {
                        states = await this.scene.dataProcessor.intersect({
                            semantic: {
                                mask: texture,
                                depth: view.depthValues,
                                depthCoverage: view.depthCoverage,
                                minDepthCoverage: 0.08,
                                depthWidth: width,
                                depthHeight: height,
                                viewMatrix: this.matrixFromArray(view.metadata.view_matrix),
                                projectionMatrix: this.matrixFromArray(view.metadata.projection_matrix),
                                near: view.metadata.near,
                                far: view.metadata.far,
                                orthographic: view.metadata.projection === 'orthographic',
                                maskThreshold: 0.5,
                                minOpacity: 0.05,
                                baseDepthTolerance: 0.0025,
                                scaleDepthTolerance: 2,
                                maxDepthTolerance: 0.05
                            }
                        }, splat);
                        for (let index = 0; index < count; index++) {
                            const state = states[index];
                            if ((state & SemanticIntersectState.LowOpacity) !== 0) continue;
                            const hit = (state & SemanticIntersectState.MaskHit) !== 0;
                            if (hit) maskHit[index]++;
                            if ((state & SemanticIntersectState.Visible) !== 0) {
                                visible[index]++;
                                if (hit) positive[index]++;
                                else visibleOutside[index]++;
                            } else if ((state & SemanticIntersectState.FrontMismatch) !== 0) {
                                frontMismatch[index]++;
                            } else if ((state & SemanticIntersectState.BehindMismatch) !== 0) {
                                behindMismatch[index]++;
                            }
                        }
                    } finally {
                        texture.destroy();
                        if (states) this.scene.dataProcessor.releaseMask(states);
                    }
                }

                const indices: number[] = [];
                const expansionCandidates: number[] = [];
                const union = combined.get(splat);
                for (let index = 0; index < count; index++) {
                    if ((splat.state.data[index] & (State.locked | State.deleted)) !== 0) continue;
                    const accepted = (positive[index] >= 2 && positive[index] > visibleOutside[index]) ||
                        (positive[index] === 1 && visible[index] === 1 && frontMismatch[index] === 0);
                    if (accepted) {
                        union[index] = 255;
                        indices.push(index);
                    } else if (
                        maskHit[index] > 0 &&
                        visibleOutside[index] === 0 &&
                        frontMismatch[index] === 0 &&
                        (behindMismatch[index] > 0 || visible[index] === 0)
                    ) {
                        expansionCandidates.push(index);
                    }
                }
                projected.push({
                    splat,
                    indices: Uint32Array.from(indices),
                    expansionCandidates: Uint32Array.from(expansionCandidates),
                    expansionApplied: false
                });
            }
            this.projectedInstances.set(instance.instance_id, projected);
        }

        let selectedCount = 0;
        for (const [splat, mask] of combined) {
            selectedCount += mask.reduce((sum, value) => sum + (value === 255 ? 1 : 0), 0);
            await this.events.invoke('select.commit', splat, 'set', mask);
            this.events.fire('selection', splat);
        }
        return selectedCount;
    }

    private async projectTo3D() {
        this.setBusy(true, '正在投射到 3D Gaussian');
        try {
            let instances: Array<{ instance_id: string; view_masks: SemanticViewMask[] }>;
            let views: CapturedView[];
            const selected = this.semanticInstances.filter(item => item.selected);
            if (selected.length && this.semanticViews.length) {
                instances = selected.map(item => ({
                    instance_id: item.instance_id,
                    view_masks: item.view_masks?.length ? item.view_masks : [{ view_index: 0, mask_url: item.mask_url }]
                }));
                views = this.semanticViews;
            } else if (this.promptView && this.promptMaskUrl) {
                instances = [{
                    instance_id: this.sessionId ?? 'prompt',
                    view_masks: [{ view_index: 0, mask_url: this.promptMaskUrl }]
                }];
                views = [this.promptView];
            } else {
                throw new Error('当前没有可投射的 mask');
            }
            const selectedCount = await this.projectInstancesTo3D(instances, views);
            this.setStatus(selectedCount ? `已投射 ${selectedCount} 个 Gaussian` : '投射结果为空，请修正 mask 或视角');
        } catch (error) {
            this.setStatus(error instanceof Error ? error.message : '3D 投射失败');
        } finally {
            this.setBusy(false);
        }
    }

    private buildExpansionGeometry(splat: Splat) {
        const count = splat.splatData.numSplats;
        const worldPositions = new Float32Array(count * 3);
        const geoScales = new Float32Array(count);
        const scale0 = splat.splatData.getProp('scale_0') as Float32Array;
        const scale1 = splat.splatData.getProp('scale_1') as Float32Array;
        const scale2 = splat.splatData.getProp('scale_2') as Float32Array;
        const transformIndices = splat.transformTexture.getSource() as unknown as ArrayLike<number>;
        const transformScale = new Map<number, number>();
        const position = new Vec3();
        const matrix = new Mat4();
        const paletteMatrix = new Mat4();
        const axisScale = new Vec3();
        const fallbackScale = Math.max(this.scene.camera.sceneRadius * 1e-6, 1e-8);

        const getTransformScale = (transformIndex: number) => {
            const cached = transformScale.get(transformIndex);
            if (cached !== undefined) return cached;
            if (transformIndex > 0) {
                splat.transformPalette.getTransform(transformIndex, paletteMatrix);
                matrix.mul2(splat.worldTransform, paletteMatrix);
            } else {
                matrix.copy(splat.worldTransform);
            }
            matrix.getScale(axisScale);
            const value = Math.cbrt(Math.abs(axisScale.x * axisScale.y * axisScale.z));
            const safeValue = Number.isFinite(value) && value > 0 ? value : 1;
            transformScale.set(transformIndex, safeValue);
            return safeValue;
        };

        for (let index = 0; index < count; index++) {
            if (splat.calcSplatWorldPosition(index, position) &&
                Number.isFinite(position.x) && Number.isFinite(position.y) && Number.isFinite(position.z)) {
                worldPositions[index * 3] = position.x;
                worldPositions[index * 3 + 1] = position.y;
                worldPositions[index * 3 + 2] = position.z;
            }
            const localScale = scale0 && scale1 && scale2 ?
                Math.exp((scale0[index] + scale1[index] + scale2[index]) / 3) :
                fallbackScale;
            const paletteIndex = transformIndices?.[index] ?? 0;
            const scale = localScale * getTransformScale(paletteIndex);
            geoScales[index] = Number.isFinite(scale) && scale > 0 ? scale : fallbackScale;
        }
        return { worldPositions, geoScales };
    }

    private buildRefinementGeometry(splat: Splat) {
        const { worldPositions, geoScales } = this.buildExpansionGeometry(splat);
        const count = splat.splatData.numSplats;
        const result = new Float32Array(count * 8);
        const red = splat.splatData.getProp('f_dc_0') as Float32Array;
        const green = splat.splatData.getProp('f_dc_1') as Float32Array;
        const blue = splat.splatData.getProp('f_dc_2') as Float32Array;
        const opacity = splat.splatData.getProp('opacity') as Float32Array;
        for (let index = 0; index < count; index++) {
            const offset = index * 8;
            result[offset] = worldPositions[index * 3];
            result[offset + 1] = worldPositions[index * 3 + 1];
            result[offset + 2] = worldPositions[index * 3 + 2];
            result[offset + 3] = Math.max(0, Math.min(1, 0.5 + (red?.[index] ?? 0) * SH_C0));
            result[offset + 4] = Math.max(0, Math.min(1, 0.5 + (green?.[index] ?? 0) * SH_C0));
            result[offset + 5] = Math.max(0, Math.min(1, 0.5 + (blue?.[index] ?? 0) * SH_C0));
            result[offset + 6] = geoScales[index];
            result[offset + 7] = opacity ? sigmoid(opacity[index]) * splat.transparency : 1;
        }
        return result;
    }

    private async refine3DSelection() {
        if (!this.projectedInstances.size) {
            this.setStatus('请先投射到 3D');
            return;
        }
        const selectedInstances = this.semanticInstances.length ?
            this.semanticInstances.filter(item => item.selected) : [];
        const semanticIds = selectedInstances.length ?
            selectedInstances.map(item => item.instance_id) : Array.from(this.projectedInstances.keys());
        if (!semanticIds.length) {
            this.setStatus('至少选择一个识别结果');
            return;
        }

        this.setBusy(true, '正在进行精细几何补全');
        try {
            const taskId = new URLSearchParams(location.search).get('task_id');
            if (!taskId) throw new Error('当前模型缺少 task_id');
            const geometryCache = new Map<Splat, Float32Array>();
            const sceneSplats = this.scene.getElementsByType(ElementType.splat) as Splat[];
            let seedCount = 0;
            let addedCount = 0;
            let engine = '';
            for (const instanceId of semanticIds) {
                const category = this.semanticInstances.find(item => item.instance_id === instanceId)?.category ?? 'object';
                for (const projected of this.projectedInstances.get(instanceId) ?? []) {
                    const clean = (index: number) =>
                        (projected.splat.state.data[index] & (State.locked | State.deleted)) === 0;
                    const seeds = Array.from(projected.indices).filter(clean);
                    const candidates = Array.from(projected.expansionCandidates).filter(clean);
                    if (!seeds.length) throw new Error('当前投射已被分离或删除，无法精细补全');
                    let geometry = geometryCache.get(projected.splat);
                    if (!geometry) {
                        geometry = this.buildRefinementGeometry(projected.splat);
                        geometryCache.set(projected.splat, geometry);
                    }
                    const sourceIndex = sceneSplats.indexOf(projected.splat);
                    if (sourceIndex < 0) throw new Error('投射源模型已离开场景');
                    const form = new FormData();
                    form.append('metadata', JSON.stringify({
                        task_id: taskId,
                        instance_id: instanceId,
                        category,
                        source_index: sourceIndex,
                        source_vertex_count: projected.splat.splatData.numSplats,
                        scene_radius: this.scene.camera.sceneRadius,
                        seed_indices: seeds,
                        candidate_indices: candidates
                    }));
                    const geometryBytes = new Uint8Array(geometry.byteLength);
                    geometryBytes.set(new Uint8Array(geometry.buffer, geometry.byteOffset, geometry.byteLength));
                    form.append('geometry', new Blob([geometryBytes.buffer], { type: 'application/octet-stream' }), 'geometry.f32');
                    const response = await fetch('/api/semantic/refine3d', { method: 'POST', body: form });
                    const body = await response.json();
                    if (!response.ok) throw new Error(body.detail ?? '精细补全服务失败');
                    projected.indices = Uint32Array.from(body.indices);
                    projected.expansionApplied = true;
                    seedCount += body.seed_count;
                    addedCount += body.added_count;
                    engine = body.engine;
                }
            }
            const combined = new Map(sceneSplats.map(splat => [splat, new Uint8Array(splat.splatData.numSplats)]));
            for (const instanceId of semanticIds) {
                for (const projected of this.projectedInstances.get(instanceId) ?? []) {
                    const mask = combined.get(projected.splat);
                    projected.indices.forEach(index => { mask[index] = 255; });
                }
            }
            for (const [splat, mask] of combined) {
                await this.events.invoke('select.commit', splat, 'set', mask);
                if (mask.some(value => value !== 0)) this.events.fire('selection', splat);
            }
            this.setStatus(`精细种子 ${seedCount}，${engine} +${addedCount}`);
        } catch (error) {
            this.setStatus(error instanceof Error ? error.message : '精细补全失败');
        } finally {
            this.setBusy(false);
        }
    }

    private async expand3DSelection() {
        if (!this.projectedInstances.size) {
            this.setStatus('请先投射到 3D');
            return;
        }
        const semanticIds = this.semanticInstances.length ?
            this.semanticInstances.filter(item => item.selected).map(item => item.instance_id) :
            Array.from(this.projectedInstances.keys());
        if (!semanticIds.length) {
            this.setStatus('至少选择一个识别结果');
            return;
        }

        this.setBusy(true, '正在补全 3D Gaussian');
        try {
            const geometry = new Map<Splat, ReturnType<SegmentationTool['buildExpansionGeometry']>>();
            let seedCount = 0;
            let addedCount = 0;
            for (const instanceId of semanticIds) {
                for (const projected of this.projectedInstances.get(instanceId) ?? []) {
                    const clean = (index: number) =>
                        (projected.splat.state.data[index] & (State.locked | State.deleted)) === 0;
                    const seeds = Array.from(projected.indices).filter(clean);
                    if (!seeds.length && projected.indices.length) {
                        throw new Error('当前投射已被分离或删除，无法继续补全');
                    }
                    const candidates = Array.from(projected.expansionCandidates).filter(clean);
                    seedCount += seeds.length;
                    let data = geometry.get(projected.splat);
                    if (!data) {
                        data = this.buildExpansionGeometry(projected.splat);
                        geometry.set(projected.splat, data);
                    }
                    const expanded = expandSemanticGaussianSeeds({
                        seedIndices: seeds,
                        candidateIndices: candidates,
                        worldPositions: data.worldPositions,
                        geoScales: data.geoScales,
                        sceneRadius: this.scene.camera.sceneRadius
                    });
                    projected.indices = expanded.indices;
                    projected.expansionApplied = true;
                    addedCount += expanded.addedCount;
                }
            }

            const splats = this.scene.getElementsByType(ElementType.splat) as Splat[];
            const combined = new Map(splats.map(splat => [splat, new Uint8Array(splat.splatData.numSplats)]));
            for (const instanceId of semanticIds) {
                for (const projected of this.projectedInstances.get(instanceId) ?? []) {
                    const mask = combined.get(projected.splat);
                    projected.indices.forEach(index => {
                        mask[index] = 255;
                    });
                }
            }
            for (const [splat, mask] of combined) {
                await this.events.invoke('select.commit', splat, 'set', mask);
                if (mask.some(value => value !== 0)) this.events.fire('selection', splat);
            }
            this.setStatus(`种子 ${seedCount}，扩张 +${addedCount}`);
        } catch (error) {
            this.setStatus(error instanceof Error ? error.message : '3D 补全失败');
        } finally {
            this.setBusy(false);
        }
    }

    private async separate3DLayer() {
        const splat = this.events.invoke('selection') as Splat;
        if (!splat || splat.numSelected === 0 || this.projectedInstances.size === 0) {
            this.setStatus('请先投射到 3D');
            return;
        }
        this.setBusy(true, '正在生成独立 3D 图层');
        try {
            const semanticName = this.semanticInstances
            .filter(item => item.selected)
            .map(item => `${item.category_zh}${item.instance_index}`)
            .join('+');
            await this.events.invoke(
                'edit.separateSelection',
                semanticName || this.nameInput.value.trim() || '分割图层'
            );
            this.setStatus('已生成独立 3D 图层，可使用撤销恢复');
        } catch (error) {
            this.setStatus(error instanceof Error ? error.message : '生成 3D 图层失败');
        } finally {
            this.setBusy(false);
        }
    }

    private buildGaussianIndexSets(instanceIds: string[]) {
        const sourceSplats = this.scene.getElementsByType(ElementType.splat) as Splat[];
        return instanceIds.flatMap((instanceId) => {
            return (this.projectedInstances.get(instanceId) ?? []).map(projected => ({
                instance_id: instanceId,
                source_index: sourceSplats.indexOf(projected.splat),
                source_vertex_count: projected.splat.splatData.numSplats,
                indices: Array.from(projected.indices)
            }));
        }).filter(item => item.source_index >= 0);
    }

    private async confirmSemantic() {
        const selectedInstances = this.semanticInstances.filter(item => item.selected);
        const selected = selectedInstances.map(item => item.instance_id);
        if (!selected.length) {
            this.setStatus('至少选择一个识别结果');
            return;
        }
        this.setBusy(true, '正在保存图层');
        try {
            if (selectedInstances.some(item => !this.projectedInstances.has(item.instance_id))) {
                const projectedCount = await this.projectInstancesTo3D(
                    selectedInstances.map(item => ({
                        instance_id: item.instance_id,
                        view_masks: item.view_masks?.length ? item.view_masks : [{ view_index: 0, mask_url: item.mask_url }]
                    })),
                    this.semanticViews
                );
                if (!projectedCount) {
                    throw new Error('3D 投射结果为空，请调整视角或识别结果');
                }
            }

            const gaussianIndexSets = this.buildGaussianIndexSets(selected);
            if (!gaussianIndexSets.some(item => item.indices.length > 0)) {
                throw new Error('没有可保存的 3D Gaussian 索引');
            }

            const response = await fetch(`/api/semantic/results/${this.semanticResultId}/confirm`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    instance_ids: selected,
                    gaussian_index_sets: gaussianIndexSets
                })
            });
            const layers = await this.readResponse(response);
            this.semanticResultId = null;
            window.alert(`已保存 ${layers.length} 个语义图层`);
            this.events.fire('tool.deactivate');
        } catch (error) {
            this.setStatus(error instanceof Error ? error.message : '保存失败');
        } finally {
            this.setBusy(false);
        }
    }

    private async readResponse(response: Response) {
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.detail || `请求失败 ${response.status}`);
        return body;
    }

    private async closeSession() {
        const sessionId = this.sessionId;
        this.sessionId = null;
        await fetch(`/api/segmentation/sessions/${sessionId}`, { method: 'DELETE' }).catch((): undefined => undefined);
    }

    private renderMarkers() {
        this.markers.replaceChildren(...this.points.map(point => {
            const marker = document.createElement('i');
            marker.className = point.label ? 'positive' : 'negative';
            marker.style.left = `${point.x * 100}%`;
            marker.style.top = `${point.y * 100}%`;
            return marker;
        }));
    }

    private clearMask() {
        this.overlay.removeAttribute('src');
        this.overlay.classList.remove('visible');
        this.promptMaskUrl = null;
        this.setStatus('单击目标添加正点');
    }

    private reset() {
        this.points = [];
        this.label = 1;
        this.renderMarkers();
        this.clearMask();
        this.semanticResultId = null;
        this.semanticInstances = [];
        this.nameInput.value = '分割图层';
        this.layerInfo.textContent = '图层：待识别';
        this.semanticViews = [];
        this.promptView = null;
        this.projectedInstances.clear();
        this.root.querySelector('.semantic-masks')?.replaceChildren();
        this.root.querySelector('.semantic-list')?.replaceChildren();
    }

    private setBusy(value: boolean, message?: string) {
        this.busy = value;
        this.root.classList.toggle('busy', value);
        this.root.querySelectorAll<HTMLButtonElement>('button').forEach(button => {
            button.disabled = value;
        });
        if (message) this.setStatus(message);
    }

    private setStatus(message: string) {
        this.status.textContent = message;
    }
}

export { SegmentationTool };
