struct PassSpatialUniform {
    source_origin_scale: vec4<f32>,
    destination_origin_scale: vec4<f32>,
    source_destination_extents: vec4<u32>,
}

struct GaussianSample {
    offset: f32,
    weight: f32,
}

@group(0) @binding(0)
var source_texture: texture_2d<f32>;

@group(0) @binding(1)
var source_sampler: sampler;

@group(0) @binding(2)
var<uniform> spatial: PassSpatialUniform;

@group(0) @binding(3)
var<storage, read> gaussian_samples: array<GaussianSample>;

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

fn source_texel(destination_position: vec2<f32>) -> vec2<f32> {
    let destination_point = spatial.destination_origin_scale.xy
        + destination_position / spatial.destination_origin_scale.z;
    return (destination_point - spatial.source_origin_scale.xy)
        * spatial.source_origin_scale.z;
}

fn sample_transparent_black(texel: vec2<f32>) -> vec4<f32> {
    let extent = vec2<f32>(spatial.source_destination_extents.xy);
    if (any(texel < vec2<f32>(0.0)) || any(texel >= extent)) {
        return vec4<f32>(0.0);
    }
    return textureSampleLevel(
        source_texture,
        source_sampler,
        texel / extent,
        0.0,
    );
}

fn blur_at(destination_position: vec2<f32>, axis: vec2<f32>) -> vec4<f32> {
    let center = source_texel(destination_position);
    var accumulated = vec4<f32>(0.0);
    for (var index = 0u; index < arrayLength(&gaussian_samples); index += 1u) {
        let sample = gaussian_samples[index];
        accumulated += sample_transparent_black(center + axis * sample.offset)
            * sample.weight;
    }
    return accumulated;
}

fn clamp_premultiplied(value: vec4<f32>) -> vec4<f32> {
    let alpha = clamp(value.a, 0.0, 1.0);
    let rgb = min(clamp(value.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(alpha));
    return vec4<f32>(rgb, alpha);
}

@fragment
fn fragment_horizontal_rgba(
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    return clamp_premultiplied(blur_at(position.xy, vec2<f32>(1.0, 0.0)));
}

@fragment
fn fragment_vertical_rgba(
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    return clamp_premultiplied(blur_at(position.xy, vec2<f32>(0.0, 1.0)));
}

@fragment
fn fragment_horizontal_source_alpha(
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    let alpha = clamp(blur_at(position.xy, vec2<f32>(1.0, 0.0)).a, 0.0, 1.0);
    return vec4<f32>(0.0, 0.0, 0.0, alpha);
}

@fragment
fn fragment_vertical_source_alpha(
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    let alpha = clamp(blur_at(position.xy, vec2<f32>(0.0, 1.0)).a, 0.0, 1.0);
    return vec4<f32>(0.0, 0.0, 0.0, alpha);
}
