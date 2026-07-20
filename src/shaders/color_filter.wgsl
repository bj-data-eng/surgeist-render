struct PassSpatialUniform {
    source_origin_scale: vec4<f32>,
    destination_origin_scale: vec4<f32>,
    source_destination_extents: vec4<u32>,
}

struct ColorFilterOperation {
    tag: u32,
    zero_flag: u32,
    exponent: i32,
    padding: u32,
    payload: vec4<f32>,
}

struct ColorFilterOperationBuffer {
    operation_count: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
    operations: array<ColorFilterOperation>,
}

@group(0) @binding(0)
var source_texture: texture_2d<f32>;

@group(0) @binding(1)
var source_sampler: sampler;

@group(0) @binding(2)
var<uniform> spatial: PassSpatialUniform;

@group(0) @binding(3)
var<storage, read> operation_buffer: ColorFilterOperationBuffer;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

fn source_uv(destination_position: vec2<f32>) -> vec2<f32> {
    let destination_point = spatial.destination_origin_scale.xy
        + destination_position / spatial.destination_origin_scale.z;
    let source_texel = (destination_point - spatial.source_origin_scale.xy)
        * spatial.source_origin_scale.z;
    return source_texel / vec2<f32>(spatial.source_destination_extents.xy);
}

fn clamp_premultiplied(value: vec4<f32>) -> vec4<f32> {
    let alpha = clamp(value.a, 0.0, 1.0);
    let rgb = min(clamp(value.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(alpha));
    return vec4<f32>(rgb, alpha);
}

fn safe_straight_rgb(premultiplied: vec4<f32>) -> vec3<f32> {
    if (premultiplied.a == 0.0) {
        return vec3<f32>(0.0);
    }
    return clamp(
        premultiplied.rgb / premultiplied.a,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn clamp_straight_then_premultiply(straight: vec4<f32>) -> vec4<f32> {
    let bounded = clamp(straight, vec4<f32>(0.0), vec4<f32>(1.0));
    return vec4<f32>(bounded.rgb * bounded.a, bounded.a);
}

fn clamp_scaled_delta(
    base_value: f32,
    delta: f32,
    operation: ColorFilterOperation,
) -> f32 {
    let base = clamp(base_value, 0.0, 1.0);
    if (operation.zero_flag != 0u || delta == 0.0) {
        return base;
    }

    let positive = delta > 0.0;
    let boundary = select(0.0, 1.0, positive);
    let distance = abs(boundary - base);
    if (distance == 0.0) {
        return boundary;
    }

    let delta_parts = frexp(abs(delta));
    var product_fraction = operation.payload.x * delta_parts.fract;
    var product_exponent = operation.exponent + delta_parts.exp;
    if (product_fraction < 0.5) {
        product_fraction = product_fraction * 2.0;
        product_exponent = product_exponent - 1;
    }

    let distance_parts = frexp(distance);
    let reaches_boundary = product_exponent > distance_parts.exp
        || (product_exponent == distance_parts.exp
            && product_fraction >= distance_parts.fract);
    if (reaches_boundary) {
        return boundary;
    }

    let magnitude = ldexp(product_fraction, product_exponent);
    let value = select(base - magnitude, base + magnitude, positive);
    return clamp(value, 0.0, 1.0);
}

fn apply_amount(
    base: vec3<f32>,
    delta: vec3<f32>,
    operation: ColorFilterOperation,
) -> vec3<f32> {
    return vec3<f32>(
        clamp_scaled_delta(base.r, delta.r, operation),
        clamp_scaled_delta(base.g, delta.g, operation),
        clamp_scaled_delta(base.b, delta.b, operation),
    );
}

fn hue_rotate(color: vec3<f32>, sine: f32, cosine: f32) -> vec3<f32> {
    return vec3<f32>(
        (0.213 + 0.787 * cosine - 0.213 * sine) * color.r
            + (0.715 - 0.715 * cosine - 0.715 * sine) * color.g
            + (0.072 - 0.072 * cosine + 0.928 * sine) * color.b,
        (0.213 - 0.213 * cosine + 0.143 * sine) * color.r
            + (0.715 + 0.285 * cosine + 0.140 * sine) * color.g
            + (0.072 - 0.072 * cosine - 0.283 * sine) * color.b,
        (0.213 - 0.213 * cosine - 0.787 * sine) * color.r
            + (0.715 - 0.715 * cosine + 0.715 * sine) * color.g
            + (0.072 + 0.928 * cosine + 0.072 * sine) * color.b,
    );
}

fn sepia_matrix(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.393 * color.r + 0.769 * color.g + 0.189 * color.b,
        0.349 * color.r + 0.686 * color.g + 0.168 * color.b,
        0.272 * color.r + 0.534 * color.g + 0.131 * color.b,
    );
}

fn apply_operation(
    premultiplied: vec4<f32>,
    operation: ColorFilterOperation,
) -> vec4<f32> {
    if (operation.tag == 5u) {
        return clamp_premultiplied(premultiplied * operation.payload.x);
    }

    var straight_rgb = safe_straight_rgb(premultiplied);
    switch operation.tag {
        case 0u: {
            straight_rgb = apply_amount(vec3<f32>(0.0), straight_rgb, operation);
        }
        case 1u: {
            straight_rgb = apply_amount(
                vec3<f32>(0.5),
                straight_rgb - vec3<f32>(0.5),
                operation,
            );
        }
        case 2u: {
            let luminance = dot(straight_rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            straight_rgb = mix(straight_rgb, vec3<f32>(luminance), operation.payload.x);
        }
        case 3u: {
            straight_rgb = hue_rotate(straight_rgb, operation.payload.x, operation.payload.y);
        }
        case 4u: {
            straight_rgb = (1.0 - operation.payload.x) * straight_rgb
                + operation.payload.x * (vec3<f32>(1.0) - straight_rgb);
        }
        case 6u: {
            let luminance = dot(straight_rgb, vec3<f32>(0.213, 0.715, 0.072));
            let base = vec3<f32>(luminance);
            straight_rgb = apply_amount(base, straight_rgb - base, operation);
        }
        case 7u: {
            straight_rgb = mix(straight_rgb, sepia_matrix(straight_rgb), operation.payload.x);
        }
        default: {}
    }
    return clamp_straight_then_premultiply(vec4<f32>(straight_rgb, premultiplied.a));
}

@fragment
fn fragment_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    var color = clamp_premultiplied(
        textureSample(source_texture, source_sampler, source_uv(position.xy)),
    );
    let count = min(
        operation_buffer.operation_count,
        arrayLength(&operation_buffer.operations),
    );
    for (var index = 0u; index < count; index = index + 1u) {
        color = apply_operation(color, operation_buffer.operations[index]);
    }
    return color;
}
