type Point3 = [number, number, number];

type RobustGeometry = {
    centerWorld: Point3;
    boundsMinWorld: Point3;
    boundsMaxWorld: Point3;
};

type SceneObject = {
    layerId: string;
    name: string;
    category: string;
    centerCamera: Point3;
    boundsMinCamera: Point3;
    boundsMaxCamera: Point3;
    screenCenter: [number, number];
    screenBounds: [number, number, number, number];
};

type SceneRelation = {
    subject: string;
    predicate: string;
    object: string;
    confidence: number;
};

const FUNCTIONS: Record<string, string[]> = {
    bottle: ['盛装液体', '储存液体', '倾倒液体'],
    cup: ['盛装饮品', '辅助饮用'],
    chair: ['供人坐下', '支撑人体'],
    bed: ['供人睡眠', '供人休息'],
    couch: ['供人坐卧', '提供休息空间'],
    tv: ['播放视听内容', '展示信息'],
    laptop: ['处理数字信息', '运行软件'],
    keyboard: ['输入文字和指令'],
    mouse: ['控制屏幕指针', '执行交互操作'],
    cell_phone: ['移动通信', '处理数字信息'],
    book: ['承载文字或图像信息', '供人阅读'],
    potted_plant: ['室内绿化', '空间装饰'],
    vase: ['盛放花枝', '空间装饰'],
    clock: ['显示时间'],
    refrigerator: ['低温储存食物', '保持食物新鲜'],
    microwave: ['加热食物'],
    oven: ['烘烤食物', '加热食物'],
    sink: ['清洗物品', '排放用水'],
    toilet: ['供人如厕'],
    dining_table: ['摆放餐具和食物', '供人用餐'],
    table: ['承载和摆放物品', '提供操作平面'],
    desk: ['提供工作或学习空间', '摆放办公用品'],
    coffee_table: ['摆放日常物品', '配合休息区域使用'],
    cabinet: ['分类收纳物品', '提供封闭储存空间'],
    wardrobe: ['收纳衣物', '储存个人用品'],
    nightstand: ['床边收纳', '摆放随手物品'],
    table_lamp: ['提供局部照明'],
    computer_monitor: ['显示数字图像和信息'],
    trash_can: ['收集废弃物'],
    door: ['连接或分隔空间', '控制人员通行'],
    window: ['提供采光', '连接室内外视野'],
    bookshelf: ['收纳和展示书籍']
};

const quantile = (sorted: number[], ratio: number) => {
    if (sorted.length === 1) return sorted[0];
    const position = (sorted.length - 1) * ratio;
    const lower = Math.floor(position);
    const fraction = position - lower;
    return sorted[lower] * (1 - fraction) + sorted[Math.min(lower + 1, sorted.length - 1)] * fraction;
};

const robustGeometry = (worldPositions: Float32Array, indices: Uint32Array): RobustGeometry => {
    const axes = [[], [], []] as number[][];
    indices.forEach((index) => {
        for (let axis = 0; axis < 3; axis++) {
            const value = worldPositions[index * 3 + axis];
            if (Number.isFinite(value)) axes[axis].push(value);
        }
    });
    if (axes.some(axis => axis.length === 0)) throw new Error('语义图层没有有效3D坐标');
    axes.forEach(axis => axis.sort((a, b) => a - b));
    return {
        centerWorld: axes.map(axis => quantile(axis, 0.5)) as Point3,
        boundsMinWorld: axes.map(axis => quantile(axis, 0.05)) as Point3,
        boundsMaxWorld: axes.map(axis => quantile(axis, 0.95)) as Point3
    };
};

const transformPoint = (matrix: ArrayLike<number>, point: Point3): Point3 => [
    matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
    matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
    matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14]
];

const projectPoint = (matrix: ArrayLike<number>, point: Point3): [number, number] => {
    const x = matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12];
    const y = matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13];
    const w = matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15];
    return Math.abs(w) > 1e-8 ? [x / w, y / w] : [x, y];
};

const corners = (min: Point3, max: Point3): Point3[] => {
    const result: Point3[] = [];
    for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) {
        result.push([x, y, z]);
    }
    return result;
};

const buildSceneObject = (
    layerId: string,
    name: string,
    category: string,
    geometry: RobustGeometry,
    viewMatrix: ArrayLike<number>,
    projectionMatrix: ArrayLike<number>
): SceneObject => {
    const cameraCorners = corners(geometry.boundsMinWorld, geometry.boundsMaxWorld)
    .map(point => transformPoint(viewMatrix, point));
    const screenCorners = cameraCorners.map(point => projectPoint(projectionMatrix, point));
    const cameraAxes = [0, 1, 2].map(axis => cameraCorners.map(point => point[axis]));
    const screenX = screenCorners.map(point => point[0]);
    const screenY = screenCorners.map(point => point[1]);
    const centerCamera = transformPoint(viewMatrix, geometry.centerWorld);
    return {
        layerId,
        name,
        category,
        centerCamera,
        boundsMinCamera: cameraAxes.map(values => Math.min(...values)) as Point3,
        boundsMaxCamera: cameraAxes.map(values => Math.max(...values)) as Point3,
        screenCenter: projectPoint(projectionMatrix, centerCamera),
        screenBounds: [Math.min(...screenX), Math.min(...screenY), Math.max(...screenX), Math.max(...screenY)]
    };
};

const analyzeRelations = (objects: SceneObject[]): SceneRelation[] => {
    const result: SceneRelation[] = [];
    const add = (a: SceneObject, predicate: string, b: SceneObject, confidence: number) => {
        result.push({ subject: a.layerId, predicate, object: b.layerId, confidence: Math.min(1, confidence) });
    };
    for (let i = 0; i < objects.length; i++) for (let j = i + 1; j < objects.length; j++) {
        const a = objects[i];
        const b = objects[j];
        const width = Math.max(0.02, ((a.screenBounds[2] - a.screenBounds[0]) + (b.screenBounds[2] - b.screenBounds[0])) / 2);
        const height = Math.max(0.02, ((a.screenBounds[3] - a.screenBounds[1]) + (b.screenBounds[3] - b.screenBounds[1])) / 2);
        const depthSize = Math.max(1e-5, ((a.boundsMaxCamera[2] - a.boundsMinCamera[2]) + (b.boundsMaxCamera[2] - b.boundsMinCamera[2])) / 2);
        const dx = a.screenCenter[0] - b.screenCenter[0];
        const dy = a.screenCenter[1] - b.screenCenter[1];
        const depthA = -a.centerCamera[2];
        const depthB = -b.centerCamera[2];
        const depthDelta = depthB - depthA;
        const horizontal = Math.abs(dx) > width * 0.25 ? (dx < 0 ? 'left' : 'right') : '';
        const longitudinal = Math.abs(depthDelta) > depthSize * 0.25 ? (depthDelta > 0 ? 'front' : 'behind') : '';
        const predicate = horizontal && longitudinal ?
            (longitudinal === 'behind' ? `${horizontal}_behind` : `${horizontal}_front_of`) :
            horizontal ? `${horizontal}_of` : longitudinal === 'front' ? 'front_of' : longitudinal;
        if (predicate) add(a, predicate, b, Math.max(Math.abs(dx) / (width * 0.25), Math.abs(depthDelta) / (depthSize * 0.25)) / 2);
        if (Math.abs(dy) > height * 0.25) add(a, dy > 0 ? 'above' : 'below', b, Math.abs(dy) / (height * 0.5));

        const gaps = [0, 1, 2].map(axis => Math.max(
            0,
            a.boundsMinCamera[axis] - b.boundsMaxCamera[axis],
            b.boundsMinCamera[axis] - a.boundsMaxCamera[axis]
        ));
        const gap = Math.hypot(...gaps);
        const diagonalA = Math.hypot(...[0, 1, 2].map(axis => a.boundsMaxCamera[axis] - a.boundsMinCamera[axis]));
        const diagonalB = Math.hypot(...[0, 1, 2].map(axis => b.boundsMaxCamera[axis] - b.boundsMinCamera[axis]));
        const screenOverlap = a.screenBounds[0] <= b.screenBounds[2] && a.screenBounds[2] >= b.screenBounds[0] &&
            a.screenBounds[1] <= b.screenBounds[3] && a.screenBounds[3] >= b.screenBounds[1];
        if (gap === 0 && screenOverlap) add(a, 'overlap', b, 1);
        else if (gap < Math.max(diagonalA, diagonalB) * 0.35) add(a, 'near', b, 1 - gap / Math.max(diagonalA, diagonalB, 1e-5));
    }
    return result;
};

const relationText = (relation: SceneRelation, names: Map<string, string>) => {
    const a = names.get(relation.subject);
    const b = names.get(relation.object);
    const labels: Record<string, string> = {
        left_of: '左侧', right_of: '右侧', front_of: '前方', behind: '后方',
        left_front_of: '左前方', right_front_of: '右前方', left_behind: '左后方', right_behind: '右后方',
        above: '上方', below: '下方'
    };
    if (relation.predicate === 'near') return `${a}与${b}彼此靠近。`;
    if (relation.predicate === 'overlap') return `${a}与${b}存在重叠。`;
    return `${a}位于${b}${labels[relation.predicate]}。`;
};

export { FUNCTIONS, analyzeRelations, buildSceneObject, relationText, robustGeometry };
export type { RobustGeometry, SceneObject, SceneRelation };
