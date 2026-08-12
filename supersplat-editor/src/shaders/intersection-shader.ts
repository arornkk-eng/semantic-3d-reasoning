const vertexShader = /* glsl */ `
    attribute vec2 vertex_position;
    void main(void) {
        gl_Position = vec4(vertex_position, 0.0, 1.0);
    }
`;

const fragmentShader = /* glsl */ `
    uniform highp usampler2D transformA;            // splat center x, y, z
    uniform sampler2D transformB;                   // exponentiated scale xyz
    uniform sampler2D splatColor;                   // gaussian color and opacity
    uniform highp usampler2D splatTransform;        // transform palette index
    uniform highp sampler2D transformPalette;       // palette of transforms
    uniform sampler2D splatState;                   // selected / locked / deleted bits
    uniform uvec2 splat_params;                     // splat texture width, num splats

    uniform mat4 matrix_model;
    uniform mat4 matrix_viewProjection;

    uniform uvec2 output_params;                    // output width, height

    // 0: mask, 1: rect, 2: sphere, 3: box, 4: semantic mask + depth
    uniform int mode;

    // mask params
    uniform sampler2D mask;                         // mask in alpha channel
    uniform vec2 mask_params;                       // mask width, height

    // semantic mask + depth params. The uploaded R32F texture stores normalized
    // linear depth in R; negative values mean missing depth.
    uniform highp sampler2D semanticDepth;
    uniform highp sampler2D semanticCoverage;
    uniform vec4 semantic_depth_params;             // width, height, near, far
    uniform mat4 semantic_viewMatrix;
    uniform vec4 semantic_params;                   // ortho, mask threshold, min opacity, transparency
    uniform vec4 semantic_tolerance;                // base, scale multiplier, max, min depth coverage

    // rect params
    uniform vec4 rect_params;                       // rect x, y, width, height

    // sphere/box params: transforms world space into the shape's local space,
    // where the shape is the unit sphere (diameter 1) or unit cube (side 1)
    uniform mat4 shape_matrix_inv;

    const uint SEMANTIC_MASK_HIT = 1u;
    const uint SEMANTIC_DEPTH_VALID = 2u;
    const uint SEMANTIC_VISIBLE = 4u;
    const uint SEMANTIC_FRONT_MISMATCH = 8u;
    const uint SEMANTIC_BEHIND_MISMATCH = 16u;
    const uint SEMANTIC_LOW_OPACITY = 32u;

    vec3 applySplatTransform(vec3 position, uint transformIndex) {
        if (transformIndex == 0u) {
            return position;
        }

        int u = int(transformIndex % 512u) * 3;
        int v = int(transformIndex / 512u);
        mat3x4 t;
        t[0] = texelFetch(transformPalette, ivec2(u, v), 0);
        t[1] = texelFetch(transformPalette, ivec2(u + 1, v), 0);
        t[2] = texelFetch(transformPalette, ivec2(u + 2, v), 0);
        return vec4(position, 1.0) * t;
    }

    void main(void) {
        // calculate output id
        uvec2 outputUV = uvec2(gl_FragCoord);
        uint outputId = (outputUV.x + outputUV.y * output_params.x) * 4u;

        vec4 clr = vec4(0.0);

        for (uint i = 0u; i < 4u; i++) {
            uint id = outputId + i;

            if (id >= splat_params.y) {
                continue;
            }

            // calculate splatUV
            ivec2 splatUV = ivec2(
                int(id % splat_params.x),
                int(id / splat_params.x)
            );

            // read splat center
            vec3 sourceCenter = uintBitsToFloat(texelFetch(transformA, splatUV, 0).xyz);

            // apply optional per-splat transform
            uint transformIndex = texelFetch(splatTransform, splatUV, 0).r;
            vec3 center = applySplatTransform(sourceCenter, transformIndex);

            // transform to world space (sphere/box modes test world-space containment)
            vec3 world = (matrix_model * vec4(center, 1.0)).xyz;

            if (mode == 0 || mode == 1 || mode == 4) {
                // screen-space modes: project to clip space and skip offscreen fragments
                vec4 clip = matrix_viewProjection * vec4(world, 1.0);
                bool semanticOrtho = mode == 4 && semantic_params.x > 0.5;
                if (mode == 4 && ((!semanticOrtho && clip.w <= 0.0) || abs(clip.w) < 1e-8)) {
                    continue;
                }
                vec3 ndc = clip.xyz / clip.w;

                if (!any(greaterThan(abs(ndc), vec3(1.0)))) {
                    if (mode == 0) {
                        // select by mask
                        ivec2 maskUV = ivec2((ndc.xy * vec2(0.5, -0.5) + 0.5) * mask_params);
                        clr[i] = texelFetch(mask, maskUV, 0).a < 1.0 ? 0.0 : 1.0;
                    } else if (mode == 1) {
                        // select by rect
                        clr[i] = all(greaterThan(ndc.xy * vec2(1.0, -1.0), rect_params.xy)) && all(lessThan(ndc.xy * vec2(1.0, -1.0), rect_params.zw)) ? 1.0 : 0.0;
                    } else {
                        // Semantic state is encoded as a byte of independent bit flags.
                        // Deleted Gaussians never participate, while locked/selected
                        // Gaussians remain valid occluders and can be classified.
                        uint vertexState = uint(texelFetch(splatState, splatUV, 0).r * 255.0 + 0.5);
                        if ((vertexState & 4u) != 0u) {
                            continue;
                        }

                        vec2 topDownUV = ndc.xy * vec2(0.5, -0.5) + 0.5;
                        ivec2 maskSize = max(ivec2(mask_params), ivec2(1));
                        ivec2 maskUV = clamp(
                            ivec2(floor(topDownUV * vec2(maskSize))),
                            ivec2(0),
                            maskSize - ivec2(1)
                        );

                        uint semanticState = 0u;
                        if (texelFetch(mask, maskUV, 0).r >= semantic_params.y) {
                            semanticState |= SEMANTIC_MASK_HIT;
                        }

                        float opacity = clamp(texelFetch(splatColor, splatUV, 0).a * semantic_params.w, 0.0, 1.0);
                        if (opacity < semantic_params.z) {
                            semanticState |= SEMANTIC_LOW_OPACITY;
                            clr[i] = float(semanticState) / 255.0;
                            continue;
                        }

                        vec4 viewPosition = semantic_viewMatrix * vec4(world, 1.0);
                        float linearDepth = -viewPosition.z;
                        float nearPlane = semantic_depth_params.z;
                        float farPlane = semantic_depth_params.w;
                        float depthRange = max(farPlane - nearPlane, 1e-6);
                        if (linearDepth < nearPlane || linearDepth > farPlane) {
                            clr[i] = float(semanticState) / 255.0;
                            continue;
                        }

                        ivec2 depthSize = max(ivec2(semantic_depth_params.xy), ivec2(1));
                        ivec2 depthUV = clamp(
                            ivec2(floor(topDownUV * vec2(depthSize))),
                            ivec2(0),
                            depthSize - ivec2(1)
                        );
                        float depthSample = texelFetch(semanticDepth, depthUV, 0).r;
                        float depthCoverage = texelFetch(semanticCoverage, depthUV, 0).r;
                        if (depthSample >= 0.0 && depthSample <= 1.0 &&
                            depthCoverage >= semantic_tolerance.w) {
                            semanticState |= SEMANTIC_DEPTH_VALID;

                            // Convert local one-sigma axes through both transform
                            // levels. Their maximum world length is a conservative
                            // depth extent for arbitrarily rotated Gaussians.
                            vec3 scale = texelFetch(transformB, splatUV, 0).xyz;
                            vec3 axisX = applySplatTransform(sourceCenter + vec3(scale.x, 0.0, 0.0), transformIndex);
                            vec3 axisY = applySplatTransform(sourceCenter + vec3(0.0, scale.y, 0.0), transformIndex);
                            vec3 axisZ = applySplatTransform(sourceCenter + vec3(0.0, 0.0, scale.z), transformIndex);
                            axisX = (matrix_model * vec4(axisX, 1.0)).xyz - world;
                            axisY = (matrix_model * vec4(axisY, 1.0)).xyz - world;
                            axisZ = (matrix_model * vec4(axisZ, 1.0)).xyz - world;
                            float maxWorldScale = max(length(axisX), max(length(axisY), length(axisZ)));

                            // Gaussian support shrinks with opacity. This is the
                            // radius, in sigmas, where opacity reaches minOpacity.
                            float opacityRadius = min(3.0, sqrt(max(0.0, 2.0 * log(opacity / semantic_params.z))));
                            float tolerance = semantic_tolerance.x +
                                semantic_tolerance.y * maxWorldScale * opacityRadius / depthRange;
                            tolerance = clamp(tolerance, semantic_tolerance.x, semantic_tolerance.z);

                            float normalizedDepth = (linearDepth - nearPlane) / depthRange;
                            float delta = normalizedDepth - depthSample;
                            if (abs(delta) <= tolerance) {
                                semanticState |= SEMANTIC_VISIBLE;
                            } else if (delta < 0.0) {
                                semanticState |= SEMANTIC_FRONT_MISMATCH;
                            } else {
                                semanticState |= SEMANTIC_BEHIND_MISMATCH;
                            }
                        }

                        clr[i] = float(semanticState) / 255.0;
                    }
                }
            } else if (mode == 2) {
                // select by sphere (world-space, independent of camera frustum):
                // unit sphere test in shape-local space
                vec3 local = (shape_matrix_inv * vec4(world, 1.0)).xyz;
                clr[i] = length(local) < 0.5 ? 1.0 : 0.0;
            } else if (mode == 3) {
                // select by box (world-space, independent of camera frustum):
                // unit cube test in shape-local space
                vec3 local = (shape_matrix_inv * vec4(world, 1.0)).xyz;
                clr[i] = all(lessThanEqual(abs(local), vec3(0.5))) ? 1.0 : 0.0;
            }
        }

        gl_FragColor = clr;
    }
`;

export { vertexShader, fragmentShader };
