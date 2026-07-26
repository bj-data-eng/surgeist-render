struct PassSpatialUniform {
    source_origin_scale: vec4<f32>,
    destination_origin_scale: vec4<f32>,
    source_destination_extents: vec4<u32>,
}

struct DropShadowParameters {
    offset: vec2<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0)
var blurred_source_alpha: texture_2d<f32>;

@group(0) @binding(1)
var blurred_source_alpha_sampler: sampler;

@group(0) @binding(2)
var<uniform> spatial: PassSpatialUniform;

@group(0) @binding(3)
var<uniform> parameters: DropShadowParameters;

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
    let unshifted_point = destination_point - parameters.offset;
    return (unshifted_point - spatial.source_origin_scale.xy)
        * spatial.source_origin_scale.z;
}

fn sample_source_alpha(destination_position: vec2<f32>) -> f32 {
    let texel = source_texel(destination_position);
    let extent = vec2<f32>(spatial.source_destination_extents.xy);
    let coverage = clamp(texel + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0))
        * clamp(extent + vec2<f32>(0.5) - texel, vec2<f32>(0.0), vec2<f32>(1.0));
    if (any(coverage == vec2<f32>(0.0))) {
        return 0.0;
    }
    let alpha = clamp(
        textureSampleLevel(
            blurred_source_alpha,
            blurred_source_alpha_sampler,
            texel / extent,
            0.0,
        ).a,
        0.0,
        1.0,
    );
    return alpha * coverage.x * coverage.y;
}

@fragment
fn fragment_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return parameters.color * sample_source_alpha(position.xy);
}
