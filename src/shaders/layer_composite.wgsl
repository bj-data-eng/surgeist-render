struct PassSpatialUniform {
    source_origin_scale: vec4<f32>,
    destination_origin_scale: vec4<f32>,
    source_destination_extents: vec4<u32>,
}

struct CompositeParameters {
    affine_linear: vec4<f32>,
    affine_translation: vec4<f32>,
    mask_bounds: vec4<f32>,
    mask_dimensions: vec4<u32>,
    mask_texel_facts: vec4<f32>,
    opacity: f32,
    blend: u32,
    mask_quality: u32,
    mask_extend: u32,
    presence: vec4<u32>,
}

@group(0) @binding(0)
var source_texture: texture_2d<f32>;

@group(0) @binding(1)
var source_sampler: sampler;

@group(0) @binding(2)
var parent_texture: texture_2d<f32>;

@group(0) @binding(3)
var clip_coverage_texture: texture_2d<f32>;

@group(0) @binding(4)
var alpha_mask_texture: texture_2d<f32>;

@group(0) @binding(5)
var<uniform> spatial: PassSpatialUniform;

@group(0) @binding(6)
var<uniform> parameters: CompositeParameters;

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

fn destination_point(destination_position: vec2<f32>) -> vec2<f32> {
    return spatial.destination_origin_scale.xy
        + destination_position / spatial.destination_origin_scale.z;
}

fn layer_local_point(destination_position: vec2<f32>) -> vec2<f32> {
    let point = destination_point(destination_position);
    return vec2<f32>(
        parameters.affine_linear.x * point.x
            + parameters.affine_linear.z * point.y
            + parameters.affine_translation.x,
        parameters.affine_linear.y * point.x
            + parameters.affine_linear.w * point.y
            + parameters.affine_translation.y,
    );
}

fn source_texel(destination_position: vec2<f32>) -> vec2<f32> {
    return (layer_local_point(destination_position) - spatial.source_origin_scale.xy)
        * spatial.source_origin_scale.z;
}

fn sample_source(destination_position: vec2<f32>) -> vec4<f32> {
    let texel = source_texel(destination_position);
    let extent = vec2<f32>(spatial.source_destination_extents.xy);
    let inside = all(texel >= vec2<f32>(0.0)) && all(texel < extent);
    let sampled = textureSample(
        source_texture,
        source_sampler,
        clamp(texel / extent, vec2<f32>(0.0), vec2<f32>(1.0)),
    );
    let bounded = select(vec4<f32>(0.0), clamp(sampled, vec4<f32>(0.0), vec4<f32>(1.0)), inside);
    let alpha = bounded.a;
    return vec4<f32>(min(bounded.rgb, vec3<f32>(alpha)), alpha);
}

fn load_parent(destination_position: vec2<f32>) -> vec4<f32> {
    let coordinate = vec2<i32>(floor(destination_position));
    let dimensions = vec2<i32>(textureDimensions(parent_texture));
    if (any(coordinate < vec2<i32>(0)) || any(coordinate >= dimensions)) {
        return vec4<f32>(0.0);
    }
    let sampled = clamp(
        textureLoad(parent_texture, coordinate, 0),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
    return vec4<f32>(min(sampled.rgb, vec3<f32>(sampled.a)), sampled.a);
}

fn sample_clip_alpha(destination_position: vec2<f32>) -> f32 {
    let coordinate = vec2<i32>(floor(destination_position));
    let dimensions = vec2<i32>(textureDimensions(clip_coverage_texture));
    if (any(coordinate < vec2<i32>(0)) || any(coordinate >= dimensions)) {
        return 0.0;
    }
    return clamp(textureLoad(clip_coverage_texture, coordinate, 0).a, 0.0, 1.0);
}

fn euclidean_mod(index: i32, length: i32) -> i32 {
    let remainder = index % length;
    return select(remainder + length, remainder, remainder >= 0);
}

fn extend_mask_index(index: i32, length: i32) -> i32 {
    if (parameters.mask_extend == 0u) {
        return clamp(index, 0, length - 1);
    }
    if (parameters.mask_extend == 1u) {
        return euclidean_mod(index, length);
    }
    let period = length * 2;
    let reflected = euclidean_mod(index, period);
    return select(period - reflected - 1, reflected, reflected < length);
}

fn mask_alpha_tap(coordinate: vec2<i32>) -> f32 {
    let dimensions = vec2<i32>(parameters.mask_dimensions.xy);
    let extended = vec2<i32>(
        extend_mask_index(coordinate.x, dimensions.x),
        extend_mask_index(coordinate.y, dimensions.y),
    );
    return textureLoad(alpha_mask_texture, extended, 0).a;
}

fn mitchell_netravali(distance_value: f32) -> f32 {
    let distance = abs(distance_value);
    let squared = distance * distance;
    let cubed = squared * distance;
    if (distance < 1.0) {
        return (7.0 * cubed - 12.0 * squared + 5.3333335) / 6.0;
    }
    if (distance < 2.0) {
        return (-2.3333335 * cubed + 12.0 * squared - 20.0 * distance + 10.666667) / 6.0;
    }
    return 0.0;
}

fn sample_mask_alpha(destination_position: vec2<f32>) -> f32 {
    let local = layer_local_point(destination_position);
    let minimum = parameters.mask_bounds.xy;
    let maximum = minimum + parameters.mask_bounds.zw;
    if (any(local < minimum) || any(local > maximum)) {
        return 0.0;
    }

    let normalized = (local - minimum) / parameters.mask_bounds.zw;
    let sample_texel = (normalized - parameters.mask_texel_facts.xy)
        / parameters.mask_texel_facts.zw;
    if (parameters.mask_quality == 0u) {
        return clamp(mask_alpha_tap(vec2<i32>(floor(sample_texel + vec2<f32>(0.5)))), 0.0, 1.0);
    }

    let base = vec2<i32>(floor(sample_texel));
    let fraction = sample_texel - vec2<f32>(base);
    if (parameters.mask_quality == 1u) {
        let top = mix(
            mask_alpha_tap(base),
            mask_alpha_tap(base + vec2<i32>(1, 0)),
            fraction.x,
        );
        let bottom = mix(
            mask_alpha_tap(base + vec2<i32>(0, 1)),
            mask_alpha_tap(base + vec2<i32>(1, 1)),
            fraction.x,
        );
        return clamp(mix(top, bottom, fraction.y), 0.0, 1.0);
    }

    var alpha = 0.0;
    for (var offset_y: i32 = -1; offset_y <= 2; offset_y = offset_y + 1) {
        let tap_y = base.y + offset_y;
        let weight_y = mitchell_netravali(sample_texel.y - f32(tap_y));
        for (var offset_x: i32 = -1; offset_x <= 2; offset_x = offset_x + 1) {
            let tap_x = base.x + offset_x;
            let weight_x = mitchell_netravali(sample_texel.x - f32(tap_x));
            alpha = alpha
                + mask_alpha_tap(vec2<i32>(tap_x, tap_y)) * weight_x * weight_y;
        }
    }
    return clamp(alpha, 0.0, 1.0);
}

fn attenuate_source(source: vec4<f32>, coverage: f32) -> vec4<f32> {
    let amount = clamp(coverage, 0.0, 1.0) * clamp(parameters.opacity, 0.0, 1.0);
    return source * amount;
}

fn source_without_outer(destination_position: vec2<f32>) -> vec4<f32> {
    return attenuate_source(sample_source(destination_position), 1.0);
}

fn source_with_clip(destination_position: vec2<f32>) -> vec4<f32> {
    return attenuate_source(
        sample_source(destination_position),
        sample_clip_alpha(destination_position),
    );
}

fn source_with_mask(destination_position: vec2<f32>) -> vec4<f32> {
    return attenuate_source(
        sample_source(destination_position),
        sample_mask_alpha(destination_position),
    );
}

fn source_with_clip_mask(destination_position: vec2<f32>) -> vec4<f32> {
    let clip_alpha = sample_clip_alpha(destination_position);
    let mask_alpha = sample_mask_alpha(destination_position);
    return attenuate_source(sample_source(destination_position), clip_alpha * mask_alpha);
}

fn safe_straight_rgb(premultiplied: vec4<f32>) -> vec3<f32> {
    if (premultiplied.a == 0.0) {
        return vec3<f32>(0.0);
    }
    return clamp(premultiplied.rgb / premultiplied.a, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn separable_blend(source: vec3<f32>, backdrop: vec3<f32>, mode: u32) -> vec3<f32> {
    switch mode {
        case 1u: {
            return source * backdrop;
        }
        case 2u: {
            return source + backdrop - source * backdrop;
        }
        case 3u: {
            let low = 2.0 * source * backdrop;
            let high = 1.0 - 2.0 * (1.0 - source) * (1.0 - backdrop);
            return select(high, low, backdrop <= vec3<f32>(0.5));
        }
        case 4u: {
            return min(source, backdrop);
        }
        case 5u: {
            return max(source, backdrop);
        }
        default: {
            return source;
        }
    }
}

fn destination_composite(source: vec4<f32>, destination_position: vec2<f32>) -> vec4<f32> {
    let backdrop = load_parent(destination_position);
    if (parameters.blend == 6u) {
        return clamp(source + backdrop, vec4<f32>(0.0), vec4<f32>(1.0));
    }

    let source_straight = safe_straight_rgb(source);
    let backdrop_straight = safe_straight_rgb(backdrop);
    let blended = separable_blend(source_straight, backdrop_straight, parameters.blend);
    let alpha = clamp(source.a + backdrop.a - source.a * backdrop.a, 0.0, 1.0);
    let premultiplied = (1.0 - backdrop.a) * source.rgb
        + (1.0 - source.a) * backdrop.rgb
        + source.a * backdrop.a * blended;
    return vec4<f32>(min(clamp(premultiplied, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(alpha)), alpha);
}

@fragment
fn fragment_normal(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return source_without_outer(position.xy);
}

@fragment
fn fragment_normal_clip(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return source_with_clip(position.xy);
}

@fragment
fn fragment_normal_mask(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return source_with_mask(position.xy);
}

@fragment
fn fragment_normal_clip_mask(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return source_with_clip_mask(position.xy);
}

@fragment
fn fragment_destination(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return destination_composite(source_without_outer(position.xy), position.xy);
}

@fragment
fn fragment_destination_clip(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return destination_composite(source_with_clip(position.xy), position.xy);
}

@fragment
fn fragment_destination_mask(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return destination_composite(source_with_mask(position.xy), position.xy);
}

@fragment
fn fragment_destination_clip_mask(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return destination_composite(source_with_clip_mask(position.xy), position.xy);
}
