type SemanticGaussianExpansionInput = {
    seedIndices: ArrayLike<number>;
    candidateIndices: ArrayLike<number>;
    worldPositions: ArrayLike<number>;
    geoScales: ArrayLike<number>;
    sceneRadius: number;
};

type SemanticGaussianExpansionResult = {
    indices: Uint32Array;
    addedCount: number;
};

type ScoredCandidate = {
    index: number;
    support: number;
};

const MAX_ROUNDS = 2;
const MAX_GROWTH_RATIO = 0.25;
const MIN_NEIGHBORS = 2;
const MIN_SUPPORT = 0.6;
const MAX_SCALE_RATIO = 4;
const MIN_RADIUS_FACTOR = 0.001;
const MAX_RADIUS_FACTOR = 0.0125;
const SCALE_RADIUS_FACTOR = 3;

const requireArrayLike = (value: ArrayLike<number>, label: string) => {
    if (!value || !Number.isInteger(value.length) || value.length < 0) {
        throw new Error(`${label} must be an array-like collection`);
    }
};

const normalizeIndices = (
    values: ArrayLike<number>,
    label: string,
    gaussianCount: number
) => {
    const unique = new Set<number>();
    for (let offset = 0; offset < values.length; offset++) {
        const index = values[offset];
        if (!Number.isInteger(index) || index < 0 || index >= gaussianCount) {
            throw new Error(`${label}[${offset}] is outside the Gaussian range: ${index}`);
        }
        unique.add(index);
    }
    return Array.from(unique).sort((a, b) => a - b);
};

const median = (values: number[]) => {
    const sorted = values.slice().sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ?
        (sorted[middle - 1] + sorted[middle]) * 0.5 :
        sorted[middle];
};

const clamp = (value: number, min: number, max: number) => {
    return Math.max(min, Math.min(max, value));
};

/**
 * Conservatively grows a fused semantic seed set through a local Gaussian graph.
 * The caller is responsible for filtering candidates using semantic evidence and
 * clean state. Output indices are unique and sorted.
 */
const expandSemanticGaussianSeeds = (
    input: SemanticGaussianExpansionInput
): SemanticGaussianExpansionResult => {
    if (!input || typeof input !== 'object') {
        throw new Error('Semantic Gaussian expansion input is required');
    }

    const {
        seedIndices,
        candidateIndices,
        worldPositions,
        geoScales,
        sceneRadius
    } = input;

    requireArrayLike(seedIndices, 'seedIndices');
    requireArrayLike(candidateIndices, 'candidateIndices');
    requireArrayLike(worldPositions, 'worldPositions');
    requireArrayLike(geoScales, 'geoScales');

    if (!Number.isFinite(sceneRadius) || sceneRadius <= 0) {
        throw new Error(`sceneRadius must be finite and positive: ${sceneRadius}`);
    }

    const gaussianCount = geoScales.length;
    if (worldPositions.length !== gaussianCount * 3) {
        throw new Error(
            `worldPositions length ${worldPositions.length} does not match ${gaussianCount} Gaussians`
        );
    }

    for (let index = 0; index < gaussianCount; index++) {
        const scale = geoScales[index];
        if (!Number.isFinite(scale) || scale <= 0) {
            throw new Error(`geoScales[${index}] must be finite and positive: ${scale}`);
        }
        const positionOffset = index * 3;
        for (let axis = 0; axis < 3; axis++) {
            const value = worldPositions[positionOffset + axis];
            if (!Number.isFinite(value)) {
                throw new Error(
                    `worldPositions[${positionOffset + axis}] must be finite: ${value}`
                );
            }
        }
    }

    const seeds = normalizeIndices(seedIndices, 'seedIndices', gaussianCount);
    const candidates = normalizeIndices(candidateIndices, 'candidateIndices', gaussianCount);
    if (seeds.length === 0 || candidates.length === 0) {
        return {
            indices: Uint32Array.from(seeds),
            addedCount: 0
        };
    }

    const minRadius = sceneRadius * MIN_RADIUS_FACTOR;
    const maxRadius = sceneRadius * MAX_RADIUS_FACTOR;
    if (!Number.isFinite(minRadius) || !Number.isFinite(maxRadius) || minRadius <= 0) {
        throw new Error(`sceneRadius produces an invalid neighbor radius: ${sceneRadius}`);
    }

    const medianSeedScale = median(seeds.map(index => geoScales[index]));
    const scaleCap = medianSeedScale * MAX_SCALE_RATIO;
    const accepted = new Uint8Array(gaussianCount);
    seeds.forEach((index) => {
        accepted[index] = 1;
    });

    const candidateSet = candidates.filter(index => accepted[index] === 0);
    if (candidateSet.length === 0) {
        return {
            indices: Uint32Array.from(seeds),
            addedCount: 0
        };
    }

    // Use the maximum possible connection radius as the cell size. Every valid
    // neighbor therefore lies in the current cell or one of its 26 neighbors.
    const nodes = Array.from(new Set([...seeds, ...candidateSet])).sort((a, b) => a - b);
    const cells = new Map<string, number[]>();
    const cellCoordinates = new Float64Array(gaussianCount * 3);
    const cellKey = (x: number, y: number, z: number) => `${x},${y},${z}`;

    nodes.forEach((index) => {
        const positionOffset = index * 3;
        const cellOffset = index * 3;
        const cellX = Math.floor(worldPositions[positionOffset] / maxRadius);
        const cellY = Math.floor(worldPositions[positionOffset + 1] / maxRadius);
        const cellZ = Math.floor(worldPositions[positionOffset + 2] / maxRadius);
        if (!Number.isSafeInteger(cellX) || !Number.isSafeInteger(cellY) || !Number.isSafeInteger(cellZ)) {
            throw new Error(`Gaussian ${index} produces an unsafe spatial hash coordinate`);
        }
        cellCoordinates[cellOffset] = cellX;
        cellCoordinates[cellOffset + 1] = cellY;
        cellCoordinates[cellOffset + 2] = cellZ;
        const key = cellKey(cellX, cellY, cellZ);
        const bucket = cells.get(key);
        if (bucket) {
            bucket.push(index);
        } else {
            cells.set(key, [index]);
        }
    });

    const scoreCandidate = (index: number): ScoredCandidate | null => {
        const positionOffset = index * 3;
        const cellOffset = index * 3;
        const x = worldPositions[positionOffset];
        const y = worldPositions[positionOffset + 1];
        const z = worldPositions[positionOffset + 2];
        const scale = geoScales[index];
        let neighborCount = 0;
        let support = 0;

        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dz = -1; dz <= 1; dz++) {
                    const bucket = cells.get(cellKey(
                        cellCoordinates[cellOffset] + dx,
                        cellCoordinates[cellOffset + 1] + dy,
                        cellCoordinates[cellOffset + 2] + dz
                    ));
                    if (!bucket) continue;

                    for (const neighbor of bucket) {
                        if (accepted[neighbor] === 0 || neighbor === index) continue;
                        const neighborScale = geoScales[neighbor];
                        const scaleRatio = Math.max(scale, neighborScale) /
                            Math.min(scale, neighborScale);
                        if (scaleRatio > MAX_SCALE_RATIO) continue;

                        const radius = clamp(
                            SCALE_RADIUS_FACTOR * (
                                Math.min(scale, scaleCap) +
                                Math.min(neighborScale, scaleCap)
                            ),
                            minRadius,
                            maxRadius
                        );
                        const neighborOffset = neighbor * 3;
                        const deltaX = x - worldPositions[neighborOffset];
                        const deltaY = y - worldPositions[neighborOffset + 1];
                        const deltaZ = z - worldPositions[neighborOffset + 2];
                        const distanceSquared = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;
                        if (distanceSquared > radius * radius) continue;

                        neighborCount++;
                        support += 1 - Math.sqrt(distanceSquared) / radius;
                    }
                }
            }
        }

        return neighborCount >= MIN_NEIGHBORS && support >= MIN_SUPPORT ?
            { index, support } :
            null;
    };

    const maxAdded = Math.ceil(seeds.length * MAX_GROWTH_RATIO);
    const added: number[] = [];
    for (let round = 0; round < MAX_ROUNDS && added.length < maxAdded; round++) {
        const scored: ScoredCandidate[] = [];
        for (const index of candidateSet) {
            if (accepted[index] !== 0) continue;
            const result = scoreCandidate(index);
            if (result) scored.push(result);
        }
        if (scored.length === 0) break;

        scored.sort((a, b) => b.support - a.support || a.index - b.index);
        const remaining = maxAdded - added.length;
        const acceptedThisRound = scored.slice(0, remaining);
        acceptedThisRound.forEach(({ index }) => {
            accepted[index] = 1;
            added.push(index);
        });
    }

    const result = [...seeds, ...added].sort((a, b) => a - b);
    return {
        indices: Uint32Array.from(result),
        addedCount: added.length
    };
};

export {
    expandSemanticGaussianSeeds,
    SemanticGaussianExpansionInput,
    SemanticGaussianExpansionResult
};
