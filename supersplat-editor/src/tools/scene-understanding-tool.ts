import { Vec3 } from 'playcanvas';

import { ElementType } from '../element';
import { Events } from '../events';
import { Scene } from '../scene';
import { Splat } from '../splat';
import { FUNCTIONS, analyzeRelations, buildSceneObject, relationText, robustGeometry } from './scene-understanding';

type IndexFile = { source_index: number; vertex_count: number; url: string };
type SemanticLayer = { layer_id: string; name: string; category?: string; gaussian_indices: IndexFile[] };
type Snapshot = {
    snapshot_id: string; name: string; description: string[];
    camera: { focal_point: number[]; azim: number; elevation: number; distance: number; fov: number };
};

class SceneUnderstandingTool {
    private events: Events;
    private scene: Scene;
    private root: HTMLDivElement;
    private layers: SemanticLayer[] = [];
    private snapshots: Snapshot[] = [];
    private selected = new Set<string>();
    private busy = false;

    constructor(events: Events, scene: Scene, host: HTMLElement) {
        this.events = events;
        this.scene = scene;
        this.root = document.createElement('div');
        this.root.id = 'scene-understanding-tool';
        this.root.innerHTML = `<div class="scene-understanding-panel">
            <header><strong>场景理解</strong><button data-action="close">×</button></header>
            <section><h4>分析对象</h4><div data-role="layers" class="understanding-list"></div></section>
            <button data-action="analyze" class="primary">分析当前视角</button>
            <section><h4>分析结果</h4><div data-role="result" class="understanding-result">请选择语义图层</div></section>
            <section><h4>已保存快照</h4><div data-role="snapshots" class="understanding-list"></div></section>
            <span data-role="status"></span></div>`;
        document.addEventListener('pointerdown', this.onDocumentPointerDown, true);
        host.appendChild(this.root);
    }

    activate = () => {
        if (!new URLSearchParams(location.search).get('task_id')) {
            window.alert('当前模型缺少 task_id，无法使用场景理解');
            this.events.fire('tool.deactivate');
            return;
        }
        this.root.classList.add('active');
        void this.refresh();
    };

    deactivate = () => this.root.classList.remove('active');

    private get taskId() {
        return new URLSearchParams(location.search).get('task_id') as string;
    }

    private async refresh() {
        this.setBusy(true, '正在加载');
        try {
            [this.layers, this.snapshots] = await Promise.all([
                this.fetchJson(`/api/tasks/${this.taskId}/layers`),
                this.fetchJson(`/api/tasks/${this.taskId}/scene-snapshots`)
            ]);
            const known = new Set(this.layers.map(layer => layer.layer_id));
            this.selected.forEach((id) => {
                if (!known.has(id)) this.selected.delete(id);
            });
            this.render();
            this.setStatus('');
        } catch (error) {
            this.setStatus(error instanceof Error ? error.message : '加载失败');
        } finally {
            this.setBusy(false);
        }
    }

    private render() {
        const layerHost = this.root.querySelector('[data-role="layers"]');
        layerHost?.replaceChildren(...this.layers.map((layer) => {
            const row = document.createElement('div');
            row.className = 'understanding-row';
            row.innerHTML = `<button class="layer-choice ${this.selected.has(layer.layer_id) ? 'selected' : ''}" data-action="toggle-layer" data-id="${layer.layer_id}">${this.selected.has(layer.layer_id) ? '☑' : '☐'} ${layer.name}</button><span><button data-action="rename-layer" data-id="${layer.layer_id}">重命名</button><button data-action="delete-layer" data-id="${layer.layer_id}">删除</button></span>`;
            return row;
        }));
        if (!this.layers.length) layerHost.textContent = '暂无已保存语义图层';
        const snapshotHost = this.root.querySelector('[data-role="snapshots"]');
        snapshotHost?.replaceChildren(...this.snapshots.map((snapshot) => {
            const row = document.createElement('div');
            row.className = 'understanding-row snapshot-row';
            row.innerHTML = `<strong>${snapshot.name}</strong><span><button data-action="view-snapshot" data-id="${snapshot.snapshot_id}">查看</button><button data-action="restore-snapshot" data-id="${snapshot.snapshot_id}">恢复视角</button><button data-action="rename-snapshot" data-id="${snapshot.snapshot_id}">重命名</button><button data-action="delete-snapshot" data-id="${snapshot.snapshot_id}">删除</button></span>`;
            return row;
        }));
        if (!this.snapshots.length) snapshotHost.textContent = '暂无快照';
    }

    private toggleLayer(id: string) {
        if (!this.selected.has(id) && this.selected.size >= 10) {
            this.setStatus('单次最多选择10个语义图层');
        } else if (this.selected.has(id)) this.selected.delete(id);
        else this.selected.add(id);
        this.render();
    }

    private async onClick(event: Event) {
        const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-action]');
        if (!button || this.busy) return;
        const { action, id } = button.dataset;
        if (action === 'close') this.events.fire('tool.deactivate');
        else if (action === 'toggle-layer') this.toggleLayer(id);
        else if (action === 'analyze') await this.analyze();
        else if (action === 'view-snapshot') this.viewSnapshot(id);
        else if (action === 'restore-snapshot') this.restoreSnapshot(id);
        else if (action === 'rename-layer') await this.renameLayer(id);
        else if (action === 'delete-layer') await this.deleteLayer(id);
        else if (action === 'rename-snapshot') await this.renameSnapshot(id);
        else if (action === 'delete-snapshot') await this.deleteSnapshot(id);
    }

    private onDocumentPointerDown = (event: PointerEvent) => {
        if (this.root.classList.contains('active') && this.root.contains(event.target as Node)) {
            void this.onClick(event);
        }
    };

    private async analyze() {
        const selectedLayers = this.layers.filter(layer => this.selected.has(layer.layer_id));
        if (!selectedLayers.length) return this.setStatus('至少选择一个语义图层');
        this.setBusy(true, '正在分析当前视角');
        try {
            const splats = this.scene.getElementsByType(ElementType.splat) as Splat[];
            const camera = this.scene.camera.camera;
            const view = Array.from(camera.viewMatrix.data);
            const projection = Array.from(camera.projectionMatrix.data);
            const objects = [];
            for (const layer of selectedLayers) {
                const packed: number[] = [];
                for (const file of layer.gaussian_indices) {
                    const splat = splats[file.source_index];
                    if (!splat || splat.splatData.numSplats !== file.vertex_count) throw new Error(`${layer.name}与当前模型不匹配`);
                    const response = await fetch(file.url);
                    if (!response.ok) throw new Error(`读取${layer.name}索引失败`);
                    const indices = new Uint32Array(await response.arrayBuffer());
                    const point = new Vec3();
                    indices.forEach((index) => {
                        if (splat.calcSplatWorldPosition(index, point)) packed.push(point.x, point.y, point.z);
                    });
                }
                const positions = Float32Array.from(packed);
                const packedIndices = Uint32Array.from({ length: positions.length / 3 }, (_, index) => index);
                objects.push(buildSceneObject(
                    layer.layer_id, layer.name, layer.category ?? 'object',
                    robustGeometry(positions, packedIndices), view, projection
                ));
            }
            const relations = analyzeRelations(objects);
            const names = new Map(objects.map(item => [item.layerId, item.name]));
            const description = relations.map(item => relationText(item, names));
            const functions: Record<string, string[]> = {};
            for (const object of objects) if (FUNCTIONS[object.category]) functions[object.category] = FUNCTIONS[object.category];
            for (const [category, uses] of Object.entries(functions)) {
                const objectNames = objects.filter(item => item.category === category).map(item => item.name).join('、');
                description.push(`${objectNames}通常用于${uses.join('、')}。`);
            }
            if (!description.length) description.push('当前选择中没有达到确认阈值的空间关系。');
            const position = this.scene.camera.position;
            const rotation = this.scene.camera.mainCamera.getRotation();
            const body = {
                camera: {
                    view_matrix: view,
                    projection_matrix: projection,
                    position: [position.x, position.y, position.z],
                    rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
                    focal_point: [this.scene.camera.focalPoint.x, this.scene.camera.focalPoint.y, this.scene.camera.focalPoint.z],
                    azim: this.scene.camera.azim,
                    elevation: this.scene.camera.elevation,
                    distance: this.scene.camera.distance,
                    fov: this.scene.camera.fov
                },
                objects: objects.map(item => ({
                    layer_id: item.layerId,
                    name: item.name,
                    category: item.category,
                    center_camera: item.centerCamera,
                    bounds_min_camera: item.boundsMinCamera,
                    bounds_max_camera: item.boundsMaxCamera
                })),
                relations: relations.map(item => ({
                    subject: item.subject, predicate: item.predicate, object: item.object, confidence: item.confidence
                })),
                functions,
                description
            };
            const snapshot = await this.fetchJson(`/api/tasks/${this.taskId}/scene-snapshots`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
            });
            this.snapshots.push(snapshot);
            this.render();
            this.showDescription(snapshot.name, description);
            this.setStatus(`已保存${snapshot.name}`);
        } catch (error) {
            this.setStatus(error instanceof Error ? error.message : '分析失败');
        } finally {
            this.setBusy(false);
        }
    }

    private viewSnapshot(id: string) {
        const item = this.snapshots.find(snapshot => snapshot.snapshot_id === id);
        if (item) this.showDescription(item.name, item.description);
    }

    private restoreSnapshot(id: string) {
        const item = this.snapshots.find(snapshot => snapshot.snapshot_id === id);
        if (!item) return;
        const camera = item.camera;
        this.scene.camera.setFocalPoint(new Vec3(camera.focal_point), 0);
        this.scene.camera.setAzimElev(camera.azim, camera.elevation, 0);
        this.scene.camera.setDistance(camera.distance, 0);
        this.scene.camera.fov = camera.fov;
        this.setStatus(`已恢复${item.name}视角`);
    }

    private async renameLayer(id: string) {
        const item = this.layers.find(layer => layer.layer_id === id);
        const name = item && window.prompt('输入新的语义图层名称', item.name)?.trim();
        if (!item || !name || name === item.name) return;
        await this.fetchJson(`/api/tasks/${this.taskId}/layers/${id}`, {
            method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name })
        });
        await this.refresh();
        this.events.fire('semantic.layersChanged');
    }

    private async deleteLayer(id: string) {
        const item = this.layers.find(layer => layer.layer_id === id);
        if (!item) return;
        const impact = await this.fetchJson(`/api/tasks/${this.taskId}/layers/${id}/delete-impact`);
        const message = `确认删除“${item.name}”？\n\n此操作将同时删除：\n• 语义图层：1个\n• 关联视角快照：${impact.snapshot_count}个\n\n原始3D模型不会被修改。`;
        if (!window.confirm(message)) return;
        await this.fetchJson(`/api/tasks/${this.taskId}/layers/${id}`, { method: 'DELETE' });
        this.selected.delete(id);
        await this.refresh();
        this.events.fire('semantic.layersChanged');
    }

    private async renameSnapshot(id: string) {
        const item = this.snapshots.find(snapshot => snapshot.snapshot_id === id);
        const name = item && window.prompt('输入新的快照名称', item.name)?.trim();
        if (!item || !name || name === item.name) return;
        await this.fetchJson(`/api/tasks/${this.taskId}/scene-snapshots/${id}`, {
            method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name })
        });
        await this.refresh();
    }

    private async deleteSnapshot(id: string) {
        const item = this.snapshots.find(snapshot => snapshot.snapshot_id === id);
        if (!item || !window.confirm(`确认删除“${item.name}”？`)) return;
        await this.fetchJson(`/api/tasks/${this.taskId}/scene-snapshots/${id}`, { method: 'DELETE' });
        await this.refresh();
    }

    private showDescription(name: string, lines: string[]) {
        const result = this.root.querySelector('[data-role="result"]');
        if (result) {
            result.replaceChildren(
                Object.assign(document.createElement('strong'), { textContent: name }),
                ...lines.map(line => Object.assign(document.createElement('p'), { textContent: line }))
            );
        }
    }

    private setBusy(value: boolean, status?: string) {
        this.busy = value;
        this.root.classList.toggle('busy', value);
        this.root.querySelectorAll<HTMLButtonElement>('button').forEach((button) => {
            button.disabled = value;
        });
        if (status) this.setStatus(status);
    }

    private setStatus(value: string) {
        const status = this.root.querySelector('[data-role="status"]');
        if (status) status.textContent = value;
    }

    private async fetchJson(url: string, init?: RequestInit) {
        const response = await fetch(url, init);
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.detail ?? `请求失败 ${response.status}`);
        return body;
    }
}

export { SceneUnderstandingTool };
