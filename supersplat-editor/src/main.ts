import { WebPCodec, WorkerQueue } from '@playcanvas/splat-transform';
import { Color, createGraphicsDevice } from 'playcanvas';

import { registerCameraPosesEvents } from './camera-poses';
import { CommandQueue } from './command-queue';
import { registerDocEvents } from './doc';
import { EditHistory } from './edit-history';
import { registerEditorEvents } from './editor';
import { ElementType } from './element';
import { Events } from './events';
import { initFileHandler } from './file-handler';
import { registerIframeApi } from './iframe-api';
import { registerPreferences } from './preferences';
import { registerPublishEvents } from './publish';
import { registerRenderEvents } from './render';
import { Scene } from './scene';
import { getSceneConfig } from './scene-config';
import { registerSelectionEvents } from './selection';
import { registerSequenceEvents } from './sequence';
import { ShortcutManager } from './shortcut-manager';
import { Splat } from './splat';
import { registerTimelineEvents } from './timeline';
import { BoxSelection } from './tools/box-selection';
import { BrushSelection } from './tools/brush-selection';
import { EyedropperSelection } from './tools/eyedropper-selection';
import { FloodSelection } from './tools/flood-selection';
import { LassoSelection } from './tools/lasso-selection';
import { MeasureTool } from './tools/measure-tool';
import { MoveTool } from './tools/move-tool';
import { OrientTool } from './tools/orient-tool';
import { PolygonSelection } from './tools/polygon-selection';
import { RealtimeDetectionOverlay } from './tools/realtime-detection-overlay';
import { RectSelection } from './tools/rect-selection';
import { RotateTool } from './tools/rotate-tool';
import { ScaleTool } from './tools/scale-tool';
import { SceneUnderstandingTool } from './tools/scene-understanding-tool';
import { SegmentationTool } from './tools/segmentation-tool';
import { SemanticLabelOverlay } from './tools/semantic-label-overlay';
import { initializeSemanticLayerSession } from './tools/semantic-layer-session';
import { SphereSelection } from './tools/sphere-selection';
import { ToolManager } from './tools/tool-manager';
import { registerTrackManagerEvents } from './track-manager';
import { registerTransformHandlerEvents } from './transform-handler';
import { BoundDimensionsOverlay } from './ui/bound-dimensions-overlay';
import { EditorUI } from './ui/editor';
import { i18n } from './ui/localization';
import { registerSelectCursor } from './ui/select-cursor';


declare global {
    interface LaunchParams {
        readonly files: FileSystemFileHandle[];
    }

    interface Window {
        launchQueue: {
            setConsumer: (callback: (launchParams: LaunchParams) => void) => void;
        };
        scene: Scene;
    }
}

const getURLArgs = () => {
    // extract settings from command line in non-prod builds only
    const config = {};

    const apply = (key: string, value: string) => {
        let obj: any = config;
        key.split('.').forEach((k, i, a) => {
            if (i === a.length - 1) {
                obj[k] = value;
            } else {
                if (!obj.hasOwnProperty(k)) {
                    obj[k] = {};
                }
                obj = obj[k];
            }
        });
    };

    const params = new URLSearchParams(window.location.search.slice(1));
    params.forEach((value: string, key: string) => {
        apply(key, value);
    });

    return config;
};

const main = async () => {
    // root events object
    const events = new Events();

    // url
    const url = new URL(window.location.href);

    // shared command queue for all async splat work (GPU readbacks + history mutations).
    // every consumer that needs ordering relative to other commands enqueues here.
    const commandQueue = new CommandQueue();

    // edit history (uses the shared queue internally)
    const editHistory = new EditHistory(events, commandQueue);

    // expose the queue as an event for any module that needs to serialise async work
    // alongside history mutations.
    events.function('queue', (fn: () => Promise<void> | void) => commandQueue.enqueue(fn));

    // init localization
    await i18n.init();

    // Configure WebP WASM for SOG format (used for both reading and writing)
    WebPCodec.wasmUrl = new URL('static/lib/webp/webp.wasm', document.baseURI).toString();

    // Run SOG writing inline rather than in worker threads. We don't ship
    // splat-transform's worker.mjs, so leaving the pool enabled makes it try to
    // spawn a worker that 404s; under SOG's parallel task load it then hangs
    // instead of falling back, producing an empty export.
    WorkerQueue.maxWorkers = 0;

    // register events that only need the events object (before UI is created)
    registerTimelineEvents(events);
    registerCameraPosesEvents(events);
    registerTrackManagerEvents(events);
    registerTransformHandlerEvents(events);
    registerPublishEvents(events);
    registerIframeApi(events);

    // initialize shortcuts
    const shortcutManager = new ShortcutManager(events);
    events.function('shortcutManager', () => shortcutManager);

    // editor ui
    const editorUI = new EditorUI(events);

    // create the graphics device
    const graphicsDevice = await createGraphicsDevice(editorUI.canvas, {
        deviceTypes: ['webgl2'],
        antialias: false,
        depth: false,
        stencil: false,
        xrCompatible: false,
        powerPreference: 'high-performance'
    });

    const urlArgs = getURLArgs();

    const overrides = [
        urlArgs
    ];

    // resolve scene config
    const sceneConfig = getSceneConfig(overrides);

    // construct the manager
    const scene = new Scene(
        events,
        sceneConfig,
        editorUI.canvas,
        graphicsDevice,
        commandQueue
    );

    // colors
    const bgClr = new Color();
    const selectedClr = new Color();
    const unselectedClr = new Color();
    const lockedClr = new Color();

    const setClr = (target: Color, value: Color, event: string) => {
        if (!target.equals(value)) {
            target.copy(value);
            events.fire(event, target);
        }
    };

    const setBgClr = (clr: Color) => {
        setClr(bgClr, clr, 'bgClr');
    };
    const setSelectedClr = (clr: Color) => {
        setClr(selectedClr, clr, 'selectedClr');
    };
    const setUnselectedClr = (clr: Color) => {
        setClr(unselectedClr, clr, 'unselectedClr');
    };
    const setLockedClr = (clr: Color) => {
        setClr(lockedClr, clr, 'lockedClr');
    };

    events.on('setBgClr', (clr: Color) => {
        setBgClr(clr);
    });
    events.on('setSelectedClr', (clr: Color) => {
        setSelectedClr(clr);
    });
    events.on('setUnselectedClr', (clr: Color) => {
        setUnselectedClr(clr);
    });
    events.on('setLockedClr', (clr: Color) => {
        setLockedClr(clr);
    });

    events.function('bgClr', () => {
        return bgClr;
    });
    events.function('selectedClr', () => {
        return selectedClr;
    });
    events.function('unselectedClr', () => {
        return unselectedClr;
    });
    events.function('lockedClr', () => {
        return lockedClr;
    });

    events.on('bgClr', (clr: Color) => {
        const cnv = (v: number) => `${Math.max(0, Math.min(255, (v * 255))).toFixed(0)}`;
        document.body.style.backgroundColor = `rgba(${cnv(clr.r)},${cnv(clr.g)},${cnv(clr.b)},1)`;
    });
    events.on('selectedClr', (clr: Color) => {
        scene.forceRender = true;
    });
    events.on('unselectedClr', (clr: Color) => {
        scene.forceRender = true;
    });
    events.on('lockedClr', (clr: Color) => {
        scene.forceRender = true;
    });

    // initialize colors from application config
    const toColor = (value: { r: number, g: number, b: number, a: number }) => {
        return new Color(value.r, value.g, value.b, value.a);
    };
    setBgClr(toColor(sceneConfig.bgClr));
    setSelectedClr(toColor(sceneConfig.selectedClr));
    setUnselectedClr(toColor(sceneConfig.unselectedClr));
    setLockedClr(toColor(sceneConfig.lockedClr));

    // create the mask selection canvas
    const maskCanvas = document.createElement('canvas');
    const maskContext = maskCanvas.getContext('2d');
    maskCanvas.setAttribute('id', 'mask-canvas');
    maskContext.globalCompositeOperation = 'copy';

    const mask = {
        canvas: maskCanvas,
        context: maskContext
    };

    // Semantic layers are temporary to one editor page session. Clear leftovers
    // before any semantic UI loads and use sendBeacon for close/reload cleanup.
    await initializeSemanticLayerSession();

    // tool manager
    const toolManager = new ToolManager(events);
    toolManager.register('rectSelection', new RectSelection(events, editorUI.toolsContainer.dom));
    toolManager.register('brushSelection', new BrushSelection(events, editorUI.toolsContainer.dom, mask));
    toolManager.register('floodSelection', new FloodSelection(events, editorUI.toolsContainer.dom, mask, editorUI.canvasContainer));
    toolManager.register('polygonSelection', new PolygonSelection(events, editorUI.toolsContainer.dom, mask));
    toolManager.register('lassoSelection', new LassoSelection(events, editorUI.toolsContainer.dom, mask));
    toolManager.register('sphereSelection', new SphereSelection(events, scene, editorUI.canvasContainer, editorUI.tooltips));
    toolManager.register('boxSelection', new BoxSelection(events, scene, editorUI.canvasContainer, editorUI.tooltips));
    toolManager.register('eyedropperSelection', new EyedropperSelection(events, editorUI.toolsContainer.dom, editorUI.canvasContainer));
    toolManager.register('move', new MoveTool(events, scene));
    toolManager.register('rotate', new RotateTool(events, scene));
    toolManager.register('scale', new ScaleTool(events, scene));
    toolManager.register('measure', new MeasureTool(events, scene, editorUI.canvasContainer));
    toolManager.register('orient', new OrientTool(events, scene, editorUI.toolsContainer.dom, editorUI.canvasContainer));
    toolManager.register('segmentation', new SegmentationTool(events, scene, editorUI.canvasContainer.dom));
    toolManager.register('sceneUnderstanding', new SceneUnderstandingTool(events, scene, editorUI.canvasContainer.dom));

    const boundDimensionsOverlay = new BoundDimensionsOverlay(events, scene, editorUI.canvasContainer);
    void new RealtimeDetectionOverlay(events, scene, editorUI.canvasContainer.dom);
    void new SemanticLabelOverlay(events, scene, editorUI.canvasContainer.dom);

    editorUI.toolsContainer.dom.appendChild(maskCanvas);

    // show the active selection op (add/remove/intersect) at the cursor
    registerSelectCursor(events, editorUI.toolsContainer.dom);

    window.scene = scene;
    (window as any).__triggerRealtimeDetection = () => events.fire('camera.moved');

    // 兼容旧调用方。选择状态必须通过 SelectOp 提交，才能进入撤销历史。
    (window as any).__selectGaussians = (indices: ArrayLike<number>, explicitTarget?: Splat) => {
        const splats = scene.getElementsByType(ElementType.splat) as Splat[];
        const current = events.invoke('selection') as Splat;
        const splat = explicitTarget ??
            (splats.includes(current) ? current : (splats.length === 1 ? splats[0] : null));
        if (!splat || !splats.includes(splat)) {
            throw new Error('Gaussian selection target is missing or ambiguous');
        }

        const safeIds = Array.from(indices)
        .filter(id => Number.isSafeInteger(id) && id >= 0);
        return events.invoke('select.commit', splat, 'set', Uint32Array.from(safeIds));
    };

    // register events that need scene or other dependencies
    registerEditorEvents(events, editHistory, scene);
    registerSelectionEvents(events, scene);
    registerSequenceEvents(events, scene);
    registerDocEvents(scene, events);
    registerRenderEvents(scene, events);
    initFileHandler(scene, events, editorUI.appContainer.dom);

    // apply stored user preferences and start capturing changes to them.
    // registered after the boot-time initialization events above so they are
    // never captured as user changes.
    registerPreferences(events, sceneConfig, urlArgs);

    // load async models
    scene.start();

    // handle load params
    const loadList = url.searchParams.getAll('load');
    const filenameList = url.searchParams.getAll('filename');
    for (const [i, value] of loadList.entries()) {
        const decoded = decodeURIComponent(value);
        const filename = i < filenameList.length ?
            decodeURIComponent(filenameList[i]) :
            decoded.split('/').pop();

        await events.invoke('import', [{
            filename,
            url: decoded
        }]);
    }


    // handle OS-based file association in PWA mode
    if ('launchQueue' in window) {
        window.launchQueue.setConsumer(async (launchParams: LaunchParams) => {
            for (const file of launchParams.files) {
                await events.invoke('import', [{
                    filename: file.name,
                    contents: await file.getFile()
                }]);
            }
        });
    }
};

export { main };
