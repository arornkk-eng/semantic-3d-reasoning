import {
    ADDRESS_CLAMP_TO_EDGE,
    FILTER_NEAREST,
    PIXELFORMAT_R32F,
    PIXELFORMAT_RGBA8,
    SEMANTIC_POSITION,
    drawQuadWithShader,
    GraphicsDevice,
    Mat4,
    RenderTarget,
    ScopeSpace,
    Shader,
    ShaderUtils,
    Texture,
    BlendState
} from 'playcanvas';

import { BufferPool } from './buffer-pool';
import { packedMaskHeight, packedMaskWidth } from './histogram-config';
import { vertexShader, fragmentShader } from '../shaders/intersection-shader';
import { Splat } from '../splat';

type MaskOptions = {
    mask: Texture;
};

// Semantic mask projection with depth visibility classification. `depth` is a
// top-down, normalized linear view-depth map where 0 = near and 1 = far. NaN
// and out-of-range values are treated as missing depth.
type SemanticMaskDepthOptions = {
    semantic: {
        mask: Texture;
        depth: Float32Array;
        depthWidth: number;
        depthHeight: number;
        viewMatrix: Mat4;
        projectionMatrix: Mat4;
        near: number;
        far: number;
        orthographic?: boolean;
        maskThreshold?: number;
        minOpacity?: number;
        baseDepthTolerance?: number;
        scaleDepthTolerance?: number;
        maxDepthTolerance?: number;
    };
};

type RectOptions = {
    rect: { x1: number, y1: number, x2: number, y2: number };
};

type SphereOptions = {
    // transform mapping the unit sphere (diameter 1) to world space
    sphere: { transform: Mat4 };
};

type BoxOptions = {
    // transform mapping the unit cube (side 1) to world space
    box: { transform: Mat4 };
};

type IntersectOptions = MaskOptions | SemanticMaskDepthOptions | RectOptions | SphereOptions | BoxOptions;

// Bit flags returned for each Gaussian by SemanticMaskDepthOptions. The result
// is still the packed byte buffer used by the other intersection modes, with
// one byte per Gaussian in the first `numSplats` bytes.
enum SemanticIntersectState {
    MaskHit = 1,
    DepthValid = 2,
    Visible = 4,
    FrontMismatch = 8,
    BehindMismatch = 16,
    LowOpacity = 32
}

const shapeInvMat = new Mat4();
const identityMat = new Mat4();

const resolve = (scope: ScopeSpace, values: any) => {
    for (const key in values) {
        scope.resolve(key).setValue(values[key]);
    }
};

class Intersect {
    private device: GraphicsDevice;
    private dummyTexture: Texture;
    private dummyDepthTexture: Texture;
    private viewProjectionMat = new Mat4();
    private shader: Shader = null;
    private texture: Texture = null;
    private renderTarget: RenderTarget = null;
    private semanticDepthTexture: Texture = null;

    constructor(device: GraphicsDevice) {
        this.device = device;
        this.dummyTexture = new Texture(device, {
            width: 1,
            height: 1,
            format: PIXELFORMAT_RGBA8
        });
        this.dummyDepthTexture = new Texture(device, {
            width: 1,
            height: 1,
            format: PIXELFORMAT_R32F,
            mipmaps: false,
            minFilter: FILTER_NEAREST,
            magFilter: FILTER_NEAREST,
            addressU: ADDRESS_CLAMP_TO_EDGE,
            addressV: ADDRESS_CLAMP_TO_EDGE
        });
        const dummyDepth = this.dummyDepthTexture.lock() as Float32Array;
        dummyDepth[0] = -1;
        this.dummyDepthTexture.unlock();
    }

    private getResources(width: number, numSplats: number) {
        const { device } = this;

        if (!this.shader) {
            this.shader = ShaderUtils.createShader(device, {
                uniqueName: 'intersectByMaskShader',
                attributes: {
                    vertex_position: SEMANTIC_POSITION
                },
                vertexGLSL: vertexShader,
                fragmentGLSL: fragmentShader
            });
        }

        const resultWidth = packedMaskWidth(width);
        const resultHeight = packedMaskHeight(resultWidth, numSplats);

        if (!this.texture || this.texture.width !== resultWidth || this.texture.height !== resultHeight) {
            if (this.texture) {
                this.texture.destroy();
                this.renderTarget.destroy();
            }

            this.texture = new Texture(device, {
                name: 'intersectTexture',
                width: resultWidth,
                height: resultHeight,
                format: PIXELFORMAT_RGBA8,
                mipmaps: false,
                addressU: ADDRESS_CLAMP_TO_EDGE,
                addressV: ADDRESS_CLAMP_TO_EDGE
            });

            this.renderTarget = new RenderTarget({
                colorBuffer: this.texture,
                depth: false
            });
        }

        return {
            shader: this.shader,
            texture: this.texture,
            renderTarget: this.renderTarget
        };
    }

    private uploadSemanticDepth(depth: Float32Array, width: number, height: number): Texture {
        if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
            throw new Error('Semantic depth dimensions must be positive integers');
        }
        if (depth.length !== width * height) {
            throw new Error(`Semantic depth length ${depth.length} does not match ${width}x${height}`);
        }

        // R32F is sampled only with texelFetch. Invalid values use a finite
        // negative sentinel so upload behavior is identical across backends.
        if (!this.semanticDepthTexture ||
            this.semanticDepthTexture.width !== width ||
            this.semanticDepthTexture.height !== height) {
            this.semanticDepthTexture?.destroy();
            this.semanticDepthTexture = new Texture(this.device, {
                name: 'semanticDepth',
                width,
                height,
                format: PIXELFORMAT_R32F,
                mipmaps: false,
                minFilter: FILTER_NEAREST,
                magFilter: FILTER_NEAREST,
                addressU: ADDRESS_CLAMP_TO_EDGE,
                addressV: ADDRESS_CLAMP_TO_EDGE
            });
        }

        const target = this.semanticDepthTexture.lock() as Float32Array;
        for (let i = 0; i < depth.length; ++i) {
            const value = depth[i];
            const valid = Number.isFinite(value) && value >= 0 && value <= 1;
            target[i] = valid ? value : -1;
        }
        this.semanticDepthTexture.unlock();
        return this.semanticDepthTexture;
    }

    async run(options: IntersectOptions, splat: Splat, bufferPool: BufferPool): Promise<Uint8Array> {
        const { device } = this;
        const { scope } = device;

        const numSplats = splat.splatData.numSplats;
        const resource = splat.entity.gsplat.instance.resource as any;
        const transformA = resource.getTexture('transformA');
        const transformB = resource.getTexture('transformB');
        const splatColor = resource.getTexture('splatColor');
        const splatTransform = splat.transformTexture;
        const transformPalette = splat.transformPalette.texture;
        const splatState = splat.stateTexture;

        const semanticOptions = (options as SemanticMaskDepthOptions).semantic;

        // update view projection matrix
        if (semanticOptions) {
            this.viewProjectionMat.mul2(semanticOptions.projectionMatrix, semanticOptions.viewMatrix);
        } else {
            const camera = splat.scene.camera.camera;
            this.viewProjectionMat.mul2(camera.projectionMatrix, camera.viewMatrix);
        }

        // allocate resources
        const resources = this.getResources(transformA.width, numSplats);

        resolve(scope, {
            transformA,
            transformB,
            splatColor,
            splatTransform,
            transformPalette,
            splatState,
            splat_params: [transformA.width, numSplats],
            matrix_model: splat.entity.getWorldTransform().data,
            matrix_viewProjection: this.viewProjectionMat.data,
            output_params: [resources.texture.width, resources.texture.height],
            semanticDepth: this.dummyDepthTexture,
            semantic_depth_params: [1, 1, 0.1, 1000],
            semantic_viewMatrix: identityMat.data,
            semantic_params: [0, 0.5, 0.03, splat.transparency],
            semantic_tolerance: [0.0025, 2.0, 0.05]
        });

        if (semanticOptions) {
            const {
                depth,
                depthWidth,
                depthHeight,
                viewMatrix,
                near,
                far
            } = semanticOptions;
            if (!(far > near)) {
                throw new Error(`Semantic depth range is invalid: near=${near}, far=${far}`);
            }
            const baseTolerance = Math.max(0, semanticOptions.baseDepthTolerance ?? 0.0025);
            const maxTolerance = Math.max(baseTolerance, semanticOptions.maxDepthTolerance ?? 0.05);
            resolve(scope, {
                mode: 4,
                mask: semanticOptions.mask,
                mask_params: [semanticOptions.mask.width, semanticOptions.mask.height],
                semanticDepth: this.uploadSemanticDepth(depth, depthWidth, depthHeight),
                semantic_depth_params: [depthWidth, depthHeight, near, far],
                semantic_viewMatrix: viewMatrix.data,
                semantic_params: [
                    semanticOptions.orthographic ? 1 : 0,
                    Math.min(1, Math.max(0, semanticOptions.maskThreshold ?? 0.5)),
                    Math.min(1, Math.max(1e-6, semanticOptions.minOpacity ?? 0.03)),
                    Math.min(1, Math.max(0, splat.transparency))
                ],
                semantic_tolerance: [
                    baseTolerance,
                    Math.max(0, semanticOptions.scaleDepthTolerance ?? 2.0),
                    maxTolerance
                ]
            });
        }

        const maskOptions = options as MaskOptions;

        if (maskOptions.mask && !semanticOptions) {
            resolve(scope, {
                mode: 0,
                mask: maskOptions.mask,
                mask_params: [maskOptions.mask.width, maskOptions.mask.height]
            });
        } else if (!semanticOptions) {
            resolve(scope, {
                mask: this.dummyTexture,
                mask_params: [0, 0]
            });
        }

        const rectOptions = options as RectOptions;
        if (rectOptions.rect) {
            resolve(scope, {
                mode: 1,
                rect_params: [
                    rectOptions.rect.x1 * 2.0 - 1.0,
                    rectOptions.rect.y1 * 2.0 - 1.0,
                    rectOptions.rect.x2 * 2.0 - 1.0,
                    rectOptions.rect.y2 * 2.0 - 1.0
                ]
            });
        } else {
            resolve(scope, {
                rect_params: [0, 0, 0, 0]
            });
        }

        const sphereOptions = options as SphereOptions;
        const boxOptions = options as BoxOptions;
        if (sphereOptions.sphere) {
            shapeInvMat.copy(sphereOptions.sphere.transform).invert();
            resolve(scope, {
                mode: 2,
                shape_matrix_inv: shapeInvMat.data
            });
        } else if (boxOptions.box) {
            shapeInvMat.copy(boxOptions.box.transform).invert();
            resolve(scope, {
                mode: 3,
                shape_matrix_inv: shapeInvMat.data
            });
        } else {
            resolve(scope, {
                shape_matrix_inv: identityMat.data
            });
        }

        device.setBlendState(BlendState.NOBLEND);
        drawQuadWithShader(device, resources.renderTarget, resources.shader);

        const byteLen = resources.texture.width * resources.texture.height * 4;
        const buffer = bufferPool.acquire(byteLen);

        const data = await resources.texture.read(0, 0, resources.texture.width, resources.texture.height, {
            renderTarget: resources.renderTarget,
            data: buffer,
            immediate: false
        });

        return data as Uint8Array;
    }
}

export {
    Intersect,
    IntersectOptions,
    MaskOptions,
    SemanticMaskDepthOptions,
    SemanticIntersectState,
    RectOptions,
    SphereOptions,
    BoxOptions
};
