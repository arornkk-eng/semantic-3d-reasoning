import { Vec3 } from 'playcanvas';

import { ElementType } from '../element';
import { Events } from '../events';
import { Scene } from '../scene';
import { Splat } from '../splat';
import { FUNCTIONS } from './scene-understanding';
import { currentTaskId, readApiResponse } from './segmentation-api';

type Point3 = [number, number, number];
type IndexFile = { source_index: number; count: number; vertex_count: number; url: string };
type SemanticLayer = {
    layer_id: string;
    name: string;
    category?: string;
    category_zh?: string;
    observation_count?: number;
    gaussian_indices: IndexFile[];
    local?: boolean;
};
type Anchor = { top: Point3; center: Point3 };
type LabelEntry = {
    layer: SemanticLayer;
    color: string;
    anchor?: Anchor;
    label: HTMLButtonElement;
    card: HTMLDivElement;
    line: SVGLineElement;
    original?: Splat;
    cuboid?: Splat;
    representation?: 'original' | 'cuboid';
};

const COLORS: Record<string, string> = {
    bottle: '#38bdf8',
    cup: '#a78bfa',
    chair: '#f59e0b',
    bed: '#f472b6',
    table: '#84cc16',
    desk: '#22c55e',
    dining_table: '#65a30d',
    couch: '#fb7185',
    laptop: '#60a5fa',
    tv: '#818cf8',
    potted_plant: '#34d399',
    object: '#f97316'
};

const quantile = (values: number[], ratio: number) => {
    values.sort((a, b) => a - b);
    if (values.length === 1) return values[0];
    const position = (values.length - 1) * ratio;
    const low = Math.floor(position);
    const fraction = position - low;
    return values[low] * (1 - fraction) + values[Math.min(low + 1, values.length - 1)] * fraction;
};

const escapeHtml = (value: string) => value.replace(/[&<>'"]/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '\'': '&#39;', '"': '&quot;'
}[character]));

class SemanticLabelOverlay {
    private events: Events;
    private scene: Scene;
    private host: HTMLElement;
    private overlay: HTMLDivElement;
    private svg: SVGSVGElement;
    private panel: HTMLElement | null;
    private entries = new Map<string, LabelEntry>();
    private baselineLayerIds = new Set<string>();
    private initialized = false;
    private visible = new Set<string>();
    private occluded = new Set<string>();
    private depthBusy = false;
    private depthTimer?: number;
    private refreshTimer?: number;

    constructor(events: Events, scene: Scene, host: HTMLElement) {
        this.events = events;
        this.scene = scene;
        this.host = host;
        this.panel = document.getElementById('semantic-layer-label-list');
        this.overlay = document.createElement('div');
        this.overlay.id = 'semantic-label-overlay';
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.classList.add('semantic-label-lines');
        this.overlay.appendChild(this.svg);
        host.appendChild(this.overlay);

        this.panel?.addEventListener('click', event => void this.onPanelClick(event));
        this.panel?.addEventListener('input', event => this.onPanelInput(event));
        events.on('update', () => this.updatePositions());
        events.on('camera.moved', () => this.scheduleDepthCheck());
        events.on('scene.elementAdded', () => this.scheduleRefresh());
        events.on('semantic.layersChanged', () => void this.refresh());
        events.on('semantic.cuboidCreated', (payload: {
            splat: Splat; cuboid: Splat; original: Splat; name: string
        }) => {
            if (!payload?.splat) return;
            const id = `cuboid-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
            const layer: SemanticLayer = {
                layer_id: id,
                name: payload.name || '识别物体',
                category_zh: payload.name || '识别物体',
                gaussian_indices: [],
                local: true
            };
            const entry = this.createEntry(layer);
            entry.anchor = this.calculateSplatAnchor(payload.splat);
            entry.original = payload.original;
            entry.cuboid = payload.cuboid;
            entry.representation = 'cuboid';
            this.entries.set(id, entry);
            this.visible.add(id);
            this.renderPanel();
            this.renderOverlay();
            this.updatePositions();
        });
        events.function('semantic.showRepresentation', (name: string, mode: 'original' | 'cuboid') => {
            const entry = Array.from(this.entries.values()).find(item => item.layer.name === name);
            if (!entry?.original || !entry.cuboid) return false;
            this.setRepresentation(entry, mode);
            return true;
        });
        events.on('scene.elementRemoved', () => window.setTimeout(() => this.removeDetachedEntries(), 0));
        void this.refresh();
    }

    private scheduleRefresh() {
        window.clearTimeout(this.refreshTimer);
        this.refreshTimer = window.setTimeout((): void => void this.refresh(), 300);
    }

    private async refresh() {
        const taskId = currentTaskId();
        if (!taskId || !this.panel) return;
        try {
            const layersResponse = await fetch(`/api/tasks/${taskId}/layers`);
            const layers = await readApiResponse(layersResponse) as SemanticLayer[];
            if (!this.initialized) {
                this.baselineLayerIds = new Set(layers.map(layer => layer.layer_id));
                this.initialized = true;
                this.renderPanel();
                return;
            }
            const known = new Set(layers.map(layer => layer.layer_id));
            for (const [id, entry] of this.entries) {
                if (!entry.layer.local && !known.has(id)) {
                    entry.label.remove();
                    entry.card.remove();
                    entry.line.remove();
                    this.entries.delete(id);
                    this.visible.delete(id);
                }
            }
            for (const layer of layers) {
                if (this.baselineLayerIds.has(layer.layer_id)) continue;
                const existing = this.entries.get(layer.layer_id);
                if (existing) existing.layer = layer;
                else this.entries.set(layer.layer_id, this.createEntry(layer));
            }
            this.renderPanel();
            this.renderOverlay();
        } catch (error) {
            this.panel.textContent = error instanceof Error ? error.message : '语义图层读取失败';
        }
    }

    private createEntry(layer: SemanticLayer): LabelEntry {
        const color = COLORS[layer.category ?? 'object'] ?? COLORS.object;
        const label = document.createElement('button');
        label.className = 'semantic-3d-label';
        label.dataset.layerId = layer.layer_id;
        label.style.setProperty('--label-color', color);
        const card = document.createElement('div');
        card.className = 'semantic-label-card';
        card.dataset.layerId = layer.layer_id;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('stroke', color);
        const entry = { layer, color, label, card, line };
        label.addEventListener('pointerdown', (event) => {
            event.stopPropagation();
            this.toggleCard(entry);
        });
        card.addEventListener('pointerdown', (event) => {
            event.stopPropagation();
            const action = (event.target as HTMLElement).closest<HTMLElement>('[data-action]');
            if (action?.dataset.action === 'hide-label') this.hideLabel(entry.layer.layer_id);
        });
        this.overlay.append(label, card);
        this.svg.appendChild(line);
        return entry;
    }

    private renderPanel() {
        if (!this.panel) return;
        if (!this.entries.size) {
            this.panel.textContent = '本次打开尚未保存新语义图层';
            return;
        }
        this.panel.replaceChildren(...Array.from(this.entries.values(), (entry) => {
            const row = document.createElement('div');
            row.className = 'semantic-layer-label-row';
            row.dataset.layerId = entry.layer.layer_id;
            const action = entry.original && entry.cuboid ?
                `<button data-action="toggle-representation">${entry.representation === 'cuboid' ? '查看原貌' : '查看黑盒'}</button>` :
                '<button data-action="refresh-anchor" title="模型编辑后重新计算标签位置">刷新位置</button>';
            row.innerHTML = `<label title="显示或隐藏3D标签"><input data-action="show" type="checkbox" ${this.visible.has(entry.layer.layer_id) ? 'checked' : ''}><span>${escapeHtml(entry.layer.name)}</span></label>
                <input data-action="color" type="color" value="${entry.color}" title="标签颜色">
                ${action}`;
            return row;
        }));
    }

    private renderOverlay() {
        for (const [id, entry] of this.entries) {
            const shown = this.visible.has(id) && Boolean(entry.anchor);
            entry.label.hidden = !shown;
            entry.card.hidden = true;
            entry.line.style.display = shown ? '' : 'none';
            entry.label.textContent = entry.layer.name;
            entry.label.style.setProperty('--label-color', entry.color);
            entry.line.setAttribute('stroke', entry.color);
        }
    }

    private async onPanelClick(event: Event) {
        const target = (event.target as HTMLElement).closest<HTMLElement>('[data-action]');
        const row = target?.closest<HTMLElement>('[data-layer-id]');
        if (!target || !row) return;
        const entry = this.entries.get(row.dataset.layerId ?? '');
        if (!entry) return;
        if (target.dataset.action === 'refresh-anchor') {
            target.setAttribute('disabled', '');
            try {
                entry.anchor = await this.calculateAnchor(entry.layer);
                this.updatePositions();
            } finally {
                target.removeAttribute('disabled');
            }
        } else if (target.dataset.action === 'toggle-representation' && entry.original && entry.cuboid) {
            this.setRepresentation(entry, entry.representation === 'cuboid' ? 'original' : 'cuboid');
        }
    }

    private setRepresentation(entry: LabelEntry, mode: 'original' | 'cuboid') {
        if (!entry.original || !entry.cuboid) return;
        entry.original.visible = mode === 'original';
        entry.cuboid.visible = mode === 'cuboid';
        entry.representation = mode;
        this.scene.forceRender = true;
        this.renderPanel();
        this.updatePositions();
    }

    private removeDetachedEntries() {
        const splats = new Set(this.scene.getElementsByType(ElementType.splat) as Splat[]);
        for (const [id, entry] of this.entries) {
            if (!entry.layer.local || !entry.original || !entry.cuboid) continue;
            if (splats.has(entry.original) || splats.has(entry.cuboid)) continue;
            entry.label.remove();
            entry.card.remove();
            entry.line.remove();
            this.entries.delete(id);
            this.visible.delete(id);
        }
        this.renderPanel();
    }

    private onPanelInput(event: Event) {
        const target = event.target as HTMLInputElement;
        const row = target.closest<HTMLElement>('[data-layer-id]');
        const entry = this.entries.get(row?.dataset.layerId ?? '');
        if (!entry) return;
        if (target.dataset.action === 'show') {
            if (target.checked) {
                this.visible.add(entry.layer.layer_id);
                void this.ensureAnchor(entry);
            } else {
                this.visible.delete(entry.layer.layer_id);
                entry.card.hidden = true;
            }
            this.renderOverlay();
        } else if (target.dataset.action === 'color') {
            entry.color = target.value;
            entry.label.style.setProperty('--label-color', entry.color);
            entry.line.setAttribute('stroke', entry.color);
        }
    }

    private toggleCard(entry: LabelEntry) {
        const open = entry.card.hidden;
        for (const item of this.entries.values()) item.card.hidden = true;
        if (open) {
            this.renderCard(entry);
            entry.card.hidden = false;
            this.positionCard(entry);
        }
    }

    private hideLabel(id: string) {
        this.visible.delete(id);
        this.renderPanel();
        this.renderOverlay();
    }

    private renderCard(entry: LabelEntry) {
        const layer = entry.layer;
        const uses = FUNCTIONS[layer.category ?? ''] ?? ['通用物体'];
        const count = layer.gaussian_indices.reduce((sum, item) => sum + item.count, 0);
        entry.card.innerHTML = `<strong>${escapeHtml(layer.name)}</strong><dl>
            <dt>类别</dt><dd>${escapeHtml(layer.category_zh ?? layer.category ?? '物体')}</dd>
            <dt>通用用途</dt><dd>${escapeHtml(uses.join('、'))}</dd>
            <dt>Gaussian</dt><dd>${count.toLocaleString()}</dd>
            <dt>观察次数</dt><dd>${layer.observation_count ?? 1}</dd></dl>
            <button data-action="hide-label">隐藏标签</button>`;
    }

    private async ensureAnchor(entry: LabelEntry) {
        try {
            if (!entry.anchor) entry.anchor = await this.calculateAnchor(entry.layer);
            this.renderOverlay();
            this.updatePositions();
            this.scheduleDepthCheck();
        } catch (error) {
            this.visible.delete(entry.layer.layer_id);
            this.renderPanel();
            this.renderOverlay();
            window.alert(error instanceof Error ? error.message : '标签位置计算失败');
        }
    }

    private async calculateAnchor(layer: SemanticLayer): Promise<Anchor> {
        const splats = this.scene.getElementsByType(ElementType.splat) as Splat[];
        const points: Point3[] = [];
        const point = new Vec3();
        const limit = 50000;
        for (const file of layer.gaussian_indices) {
            const splat = splats[file.source_index];
            if (!splat || splat.splatData.numSplats !== file.vertex_count) continue;
            const response = await fetch(file.url);
            if (!response.ok) throw new Error(`读取${layer.name}索引失败`);
            const indices = new Uint32Array(await response.arrayBuffer());
            const step = Math.max(1, Math.ceil(indices.length / Math.max(1, limit - points.length)));
            for (let offset = 0; offset < indices.length && points.length < limit; offset += step) {
                if (splat.calcSplatWorldPosition(indices[offset], point)) {
                    points.push([point.x, point.y, point.z]);
                }
            }
        }
        if (!points.length) throw new Error(`${layer.name}没有可用3D坐标`);
        const center: Point3 = [0, 1, 2].map(axis => quantile(points.map(item => item[axis]), 0.5)) as Point3;
        const up: Point3 = [0, 1, 0];
        const heights = points.map(item => item[0] * up[0] + item[1] * up[1] + item[2] * up[2]);
        const centerHeight = center[0] * up[0] + center[1] * up[1] + center[2] * up[2];
        const topHeight = quantile(heights, 0.95);
        return {
            center,
            top: [center[0] + up[0] * (topHeight - centerHeight),
                center[1] + up[1] * (topHeight - centerHeight),
                center[2] + up[2] * (topHeight - centerHeight)]
        };
    }

    private calculateSplatAnchor(splat: Splat): Anchor {
        const points: Point3[] = [];
        const point = new Vec3();
        const count = splat.splatData.numSplats;
        const step = Math.max(1, Math.ceil(count / 50000));
        for (let index = 0; index < count; index += step) {
            if (splat.calcSplatWorldPosition(index, point)) {
                points.push([point.x, point.y, point.z]);
            }
        }
        if (!points.length) throw new Error('黑色长方体没有可用3D坐标');
        const center: Point3 = [0, 1, 2].map(axis => quantile(points.map(item => item[axis]), 0.5)) as Point3;
        const topY = quantile(points.map(item => item[1]), 0.95);
        return { center, top: [center[0], topY, center[2]] };
    }

    private project(point: Point3) {
        const cameraPoint = new Vec3(point[0], point[1], point[2]);
        this.scene.camera.camera.viewMatrix.transformPoint(cameraPoint, cameraPoint);
        if (cameraPoint.z >= 0) return null;
        const screen = new Vec3(point[0], point[1], point[2]);
        this.scene.camera.worldToScreen(screen, screen);
        if (screen.x < 0 || screen.x > 1 || screen.y < 0 || screen.y > 1) return null;
        return { x: screen.x * this.host.clientWidth, y: screen.y * this.host.clientHeight, depth: -cameraPoint.z };
    }

    private updatePositions() {
        const placed: DOMRect[] = [];
        for (const [id, entry] of this.entries) {
            if (!this.visible.has(id) || !entry.anchor) continue;
            const top = this.project(entry.anchor.top);
            const center = this.project(entry.anchor.center);
            if (!top || !center) {
                entry.label.hidden = true;
                entry.card.hidden = true;
                entry.line.style.display = 'none';
                continue;
            }
            entry.label.hidden = false;
            entry.line.style.display = '';
            const width = entry.label.offsetWidth || 70;
            const height = entry.label.offsetHeight || 26;
            const left = Math.max(4, Math.min(this.host.clientWidth - width - 4, top.x - width / 2));
            let y = Math.max(4, top.y - height - 18);
            for (let attempt = 0; attempt < 12; attempt++) {
                const rect = new DOMRect(left, y, width, height);
                if (!placed.some(other => rect.left < other.right + 4 && rect.right + 4 > other.left &&
                    rect.top < other.bottom + 4 && rect.bottom + 4 > other.top)) break;
                y = Math.min(this.host.clientHeight - height - 4, y + height + 6);
            }
            const rect = new DOMRect(left, y, width, height);
            placed.push(rect);
            entry.label.style.transform = `translate(${left}px, ${y}px)`;
            const opacity = this.occluded.has(id) ? '0.3' : '1';
            entry.label.style.opacity = opacity;
            entry.line.style.opacity = opacity;
            entry.line.setAttribute('x1', String(left + width / 2));
            entry.line.setAttribute('y1', String(y + height));
            entry.line.setAttribute('x2', String(center.x));
            entry.line.setAttribute('y2', String(center.y));
            if (!entry.card.hidden) this.positionCard(entry);
        }
    }

    private positionCard(entry: LabelEntry) {
        const left = parseFloat(entry.label.style.transform.match(/translate\(([-\d.]+)px/)?.[1] ?? '0');
        const y = parseFloat(entry.label.style.transform.match(/, ([-\d.]+)px/)?.[1] ?? '0');
        const cardWidth = entry.card.offsetWidth || 230;
        entry.card.style.left = `${Math.min(this.host.clientWidth - cardWidth - 6, left + entry.label.offsetWidth + 8)}px`;
        entry.card.style.top = `${Math.max(6, Math.min(this.host.clientHeight - entry.card.offsetHeight - 6, y))}px`;
    }

    private scheduleDepthCheck() {
        window.clearTimeout(this.depthTimer);
        this.depthTimer = window.setTimeout((): void => void this.checkOcclusion(), 250);
    }

    private async checkOcclusion() {
        if (this.depthBusy || !this.visible.size || this.host.clientWidth === 0) return;
        this.depthBusy = true;
        try {
            const width = 320;
            const height = Math.max(1, Math.round(width * this.host.clientHeight / this.host.clientWidth));
            const data = await this.events.invoke('render.depthData', width, height) as {
                depth: Float32Array; coverage: Float32Array
            };
            const near = this.scene.camera.near;
            const far = this.scene.camera.far;
            this.occluded.clear();
            for (const [id, entry] of this.entries) {
                if (!this.visible.has(id) || !entry.anchor) continue;
                const projected = this.project(entry.anchor.top);
                if (!projected) continue;
                const x = Math.min(width - 1, Math.max(0, Math.floor(projected.x / this.host.clientWidth * width)));
                const y = Math.min(height - 1, Math.max(0, Math.floor(projected.y / this.host.clientHeight * height)));
                const index = y * width + x;
                const expected = (projected.depth - near) / (far - near);
                if (data.coverage[index] > 0.05 && Number.isFinite(data.depth[index]) && data.depth[index] + 0.008 < expected) {
                    this.occluded.add(id);
                }
            }
            this.updatePositions();
        } finally {
            this.depthBusy = false;
        }
    }
}

export { SemanticLabelOverlay };
