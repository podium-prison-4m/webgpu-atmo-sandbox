/// <reference types="@webgpu/types" />

const PARTICLE_COUNT = 250_000;
const NOISE_SIZE = 128;

const BAKE_NOISE_WGSL = /* wgsl */ `
@group(0) @binding(0) var outTex: texture_storage_3d<rgba8unorm, write>;

fn hash3(p: vec3<f32>) -> vec3<f32> {
  var q = vec3<f32>(
    dot(p, vec3<f32>(127.1, 311.7, 74.7)),
    dot(p, vec3<f32>(269.5, 183.3, 246.1)),
    dot(p, vec3<f32>(113.5, 271.9, 124.6))
  );
  return fract(sin(q) * 43758.5453123);
}

fn tileableWorley(uv: vec3<f32>, freq: f32) -> f32 {
  let p = uv * freq;
  let id = floor(p);
  let f = fract(p);
  var minDist = 1.0;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let offset = vec3<f32>(f32(x), f32(y), f32(z));
        let cell = ((id + offset) % vec3<f32>(freq) + vec3<f32>(freq)) % vec3<f32>(freq);
        let rnd = hash3(cell);
        let diff = offset + rnd - f;
        minDist = min(minDist, dot(diff, diff));
      }
    }
  }
  return 1.0 - clamp(sqrt(minDist), 0.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= vec3<u32>(128u))) { return; }
  let uv = vec3<f32>(gid) / 128.0;
  let w1 = tileableWorley(uv, 4.0);
  let w2 = tileableWorley(uv, 8.0);
  let w3 = tileableWorley(uv, 16.0);
  let fbm = w1 * 0.625 + w2 * 0.25 + w3 * 0.125;
  textureStore(outTex, gid, vec4<f32>(fbm, w1, w2, w3));
}
`;

const PARTICLE_COMPUTE_WGSL = /* wgsl */ `
struct SimParams {
  time: f32,
  dt: f32,
  sunElevation: f32,
  cloudCoverage: f32,

  curlStrength: f32,
  mouseActive: f32,
  aspect: f32,
  _pad0: f32,

  mouseWorld: vec3<f32>,
  _pad1: f32,

  windDir: vec3<f32>,
  _pad2: f32,
};

struct Particle {
  pos: vec3<f32>,
  life: f32,
  vel: vec3<f32>,
  size: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> srcParticles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> dstParticles: array<Particle>;

fn hash13(p3: vec3<f32>) -> f32 {
  var p = fract(p3 * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}

fn noise3(x: vec3<f32>) -> f32 {
  let i = floor(x);
  let f = fract(x);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash13(i + vec3<f32>(0,0,0)), hash13(i + vec3<f32>(1,0,0)), u.x),
        mix(hash13(i + vec3<f32>(0,1,0)), hash13(i + vec3<f32>(1,1,0)), u.x), u.y),
    mix(mix(hash13(i + vec3<f32>(0,0,1)), hash13(i + vec3<f32>(1,0,1)), u.x),
        mix(hash13(i + vec3<f32>(0,1,1)), hash13(i + vec3<f32>(1,1,1)), u.x), u.y),
    u.z
  );
}

fn curlNoise(p: vec3<f32>) -> vec3<f32> {
  let e = 0.05;
  let dx = vec3<f32>(e, 0.0, 0.0);
  let dy = vec3<f32>(0.0, e, 0.0);
  let dz = vec3<f32>(0.0, 0.0, e);

  let p_x0 = noise3(p - dx); let p_x1 = noise3(p + dx);
  let p_y0 = noise3(p - dy); let p_y1 = noise3(p + dy);
  let p_z0 = noise3(p - dz); let p_z1 = noise3(p + dz);

  let x = (p_y1 - p_y0) - (p_z1 - p_z0);
  let y = (p_z1 - p_z0) - (p_x1 - p_x0);
  let z = (p_x1 - p_x0) - (p_y1 - p_y0);
  return normalize(vec3<f32>(x, y, z) + vec3<f32>(1e-5));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&srcParticles)) { return; }

  var p = srcParticles[idx];
  p.life -= params.dt * 0.18;

  if (p.life <= 0.0) {
    let seed = vec3<f32>(f32(idx) * 0.013, params.time * 1.7, f32(idx % 97u));
    p.pos = vec3<f32>(
      (hash13(seed) - 0.5) * 14.0,
      -1.8 + hash13(seed + 7.1) * 1.2,
      (hash13(seed + 19.3) - 0.5) * 10.0
    );
    p.vel = vec3<f32>(0.0, 0.2, 0.0);
    p.life = 0.6 + hash13(seed + 3.7) * 0.4;
  }

  let flowPos = p.pos * 0.35 + vec3<f32>(params.time * 0.15, -params.time * 0.08, 0.0);
  var force = curlNoise(flowPos) * params.curlStrength + params.windDir * 0.8 + vec3<f32>(0.0, 0.25, 0.0);

  if (params.mouseActive > 0.5) {
    let toMouse = params.mouseWorld - p.pos;
    let dist = length(toMouse);
    if (dist < 3.5 && dist > 0.05) {
      let falloff = 1.0 - dist / 3.5;
      let radial = toMouse / dist;
      let tangent = cross(vec3<f32>(0.0, 0.0, 1.0), radial);
      force += (tangent * 9.0 + radial * 2.5 + vec3<f32>(0.0, 3.0, 0.0)) * falloff * falloff;
    }
  }

  p.vel = mix(p.vel, force, clamp(params.dt * 3.5, 0.0, 1.0));
  p.pos += p.vel * params.dt;

  dstParticles[idx] = p;
}
`;

const RENDER_WGSL = /* wgsl */ `
struct SimParams {
  time: f32,
  dt: f32,
  sunElevation: f32,
  cloudCoverage: f32,

  curlStrength: f32,
  mouseActive: f32,
  aspect: f32,
  _pad0: f32,

  mouseWorld: vec3<f32>,
  _pad1: f32,

  windDir: vec3<f32>,
  _pad2: f32,
};

struct Particle {
  pos: vec3<f32>,
  life: f32,
  vel: vec3<f32>,
  size: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var cloudTex: texture_3d<f32>;
@group(0) @binding(2) var cloudSampler: sampler;

struct SkyVSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vsSky(@builtin(vertex_index) vid: u32) -> SkyVSOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  var out: SkyVSOut;
  out.pos = vec4<f32>(pos[vid], 0.999, 1.0);
  out.uv = pos[vid];
  return out;
}

fn hgPhase(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 0.001), 1.5));
}

fn computeSkyColor(rd: vec3<f32>, sunDir: vec3<f32>) -> vec3<f32> {
  let zenith = max(rd.y, 0.001);
  let sunCos = dot(rd, sunDir);
  let sunHeight = clamp(sunDir.y, -0.2, 1.0);

  let dayZenith = vec3<f32>(0.12, 0.36, 0.78);
  let dayHorizon = vec3<f32>(0.55, 0.75, 0.95);
  let sunsetHorizon = vec3<f32>(0.98, 0.42, 0.16);
  let nightColor = vec3<f32>(0.015, 0.025, 0.05);

  let dayFactor = smoothstep(-0.15, 0.35, sunHeight);
  let sunsetFactor = exp(-abs(sunHeight - 0.08) * 9.0);

  var sky = mix(dayHorizon, dayZenith, pow(zenith, 0.45)) * dayFactor;
  sky = mix(sky, sunsetHorizon * (0.4 + 0.6 * pow(max(sunCos, 0.0), 2.0)), sunsetFactor * (1.0 - zenith * 0.7));
  sky = max(sky, nightColor);

  let mie = hgPhase(sunCos, 0.76) * 0.35 * max(dayFactor, 0.1);
  let sunDisk = smoothstep(0.9985, 0.9996, sunCos) * 18.0 * max(dayFactor, 0.05);
  let sunTint = mix(vec3<f32>(1.0, 0.45, 0.18), vec3<f32>(1.0, 0.96, 0.88), smoothstep(0.0, 0.35, sunHeight));

  return sky + sunTint * (mie + sunDisk);
}

fn sampleCloudDensity(p: vec3<f32>) -> f32 {
  let hFrac = clamp((p.y - 1.2) / 2.2, 0.0, 1.0);
  let heightProfile = smoothstep(0.0, 0.2, hFrac) * (1.0 - smoothstep(0.65, 1.0, hFrac));
  if (heightProfile <= 0.001) { return 0.0; }

  let uvw = p * vec3<f32>(0.08, 0.14, 0.08) + vec3<f32>(params.time * 0.012, 0.0, params.time * 0.006);
  let n = textureSampleLevel(cloudTex, cloudSampler, uvw, 0.0);
  let baseShape = n.r * heightProfile;
  return max(baseShape - (1.0 - params.cloudCoverage), 0.0) * 2.6;
}

@fragment
fn fsSky(in: SkyVSOut) -> @location(0) vec4<f32> {
  let ro = vec3<f32>(0.0, 0.0, 8.0);
  let rd = normalize(vec3<f32>(in.uv.x * params.aspect * 0.75, in.uv.y * 0.75, -1.0));
  let sunDir = normalize(vec3<f32>(0.45, params.sunElevation, -0.65));

  var col = computeSkyColor(rd, sunDir);

  if (rd.y > 0.02) {
    let tMin = 1.2 / rd.y;
    let tMax = min(3.4 / rd.y, 26.0);
    let steps = 28;
    let dt = (tMax - tMin) / f32(steps);

    var transmittance = 1.0;
    var scatteredLight = vec3<f32>(0.0);
    let cosTheta = dot(rd, sunDir);

    let sunLight = mix(vec3<f32>(1.0, 0.42, 0.15), vec3<f32>(1.35, 1.28, 1.15), smoothstep(-0.05, 0.35, sunDir.y));
    let ambientLight = mix(vec3<f32>(0.05, 0.07, 0.12), vec3<f32>(0.42, 0.55, 0.78), smoothstep(-0.1, 0.3, sunDir.y));

    var t = tMin;
    for (var i = 0; i < steps; i++) {
      if (transmittance < 0.01) { break; }

      let pos = ro + rd * t;
      let dens = sampleCloudDensity(pos);
      if (dens > 0.001) {
        let lightDens = sampleCloudDensity(pos + sunDir * 0.35) + sampleCloudDensity(pos + sunDir * 0.85) * 0.5;

        var multiScatter = 0.0;
        var atten = 1.0;
        var phaseAtten = 1.0;
        for (var oct = 0; oct < 3; oct++) {
          let beer = exp(-lightDens * 1.4 * atten);
          let powder = 1.0 - exp(-lightDens * 2.8 * atten);
          let phase = mix(hgPhase(cosTheta, -0.25 * phaseAtten), hgPhase(cosTheta, 0.72 * phaseAtten), 0.75);
          multiScatter += beer * mix(1.0, powder, 0.45) * phase * atten;
          atten *= 0.5;
          phaseAtten *= 0.6;
        }

        let stepSigma = dens * 1.35;
        let stepTrans = exp(-stepSigma * dt);
        let luminance = sunLight * multiScatter * 2.4 + ambientLight;

        scatteredLight += transmittance * luminance * (1.0 - stepTrans);
        transmittance *= stepTrans;
      }
      t += dt;
    }
    col = col * transmittance + scatteredLight;
  } else {
    let groundFog = mix(vec3<f32>(0.03, 0.04, 0.06), vec3<f32>(0.38, 0.48, 0.58), smoothstep(-0.1, 0.3, sunDir.y));
    col = mix(col, groundFog, smoothstep(0.02, -0.15, rd.y));
  }

  col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
  col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
  return vec4<f32>(col, 1.0);
}

@group(0) @binding(3) var<storage, read> particles: array<Particle>;

struct ParticleVSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) quadUV: vec2<f32>,
  @location(1) alpha: f32,
  @location(2) color: vec3<f32>,
};

@vertex
fn vsParticle(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> ParticleVSOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>( 1.0, 1.0)
  );
  let p = particles[iid];
  let c = corners[vid];

  let relPos = p.pos - vec3<f32>(0.0, 0.0, 8.0);
  let zDist = max(-relPos.z, 0.5);
  let scale = 1.0 / (zDist * 0.75);
  let ndc = vec2<f32>((relPos.x * scale) / params.aspect, relPos.y * scale);
  let size = p.size * scale;

  let sunTint = mix(
    vec3<f32>(0.95, 0.45, 0.2),
    vec3<f32>(0.92, 0.85, 0.72),
    smoothstep(-0.05, 0.35, params.sunElevation)
  );
  let speedGlow = clamp(length(p.vel) * 0.22, 0.0, 1.0);

  var out: ParticleVSOut;
  out.pos = vec4<f32>(ndc + c * vec2<f32>(size / params.aspect, size), 0.5, 1.0);
  out.quadUV = c;
  out.alpha = smoothstep(0.0, 0.15, p.life) * (1.0 - smoothstep(0.75, 1.0, p.life)) * 0.42;
  out.color = mix(sunTint, vec3<f32>(0.45, 0.82, 1.0), speedGlow * 0.55);
  return out;
}

@fragment
fn fsParticle(in: ParticleVSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.quadUV, in.quadUV);
  if (r2 > 1.0) { discard; }
  let soft = exp(-r2 * 3.2) * in.alpha;
  return vec4<f32>(in.color * soft, soft);
}
`;

async function init() {
  if (!navigator.gpu) {
    alert("Current browser does not support WebGPU. Please use Chrome/Edge 113+.");
    return;
  }

  const canvas = document.getElementById("gfx") as HTMLCanvasElement;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter found");
  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu")!;
  const format = navigator.gpu.getPreferredCanvasFormat();

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    context.configure({ device, format, alphaMode: "opaque" });
  }
  window.addEventListener("resize", resize);
  resize();

  const cloudTexture = device.createTexture({
    size: [NOISE_SIZE, NOISE_SIZE, NOISE_SIZE],
    dimension: "3d",
    format: "rgba8unorm",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  const bakeModule = device.createShaderModule({ code: BAKE_NOISE_WGSL });
  const bakePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: bakeModule, entryPoint: "main" },
  });
  const bakeBindGroup = device.createBindGroup({
    layout: bakePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: cloudTexture.createView() }],
  });

  {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(bakePipeline);
    pass.setBindGroup(0, bakeBindGroup);
    pass.dispatchWorkgroups(NOISE_SIZE / 4, NOISE_SIZE / 4, NOISE_SIZE / 4);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  const initialParticles = new Float32Array(PARTICLE_COUNT * 8);
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const o = i * 8;
    initialParticles[o + 0] = (Math.random() - 0.5) * 14.0;
    initialParticles[o + 1] = (Math.random() - 0.5) * 5.0;
    initialParticles[o + 2] = (Math.random() - 0.5) * 10.0;
    initialParticles[o + 3] = Math.random();
    initialParticles[o + 4] = 0;
    initialParticles[o + 5] = 0.2;
    initialParticles[o + 6] = 0;
    initialParticles[o + 7] = 0.012 + Math.random() * 0.018;
  }

  const particleBuffers = [0, 1].map(() => {
    const buf = device.createBuffer({
      size: initialParticles.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, initialParticles);
    return buf;
  });

  const simParamsData = new Float32Array(16);
  const simParamsBuffer = device.createBuffer({
    size: simParamsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const computeModule = device.createShaderModule({ code: PARTICLE_COMPUTE_WGSL });
  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: computeModule, entryPoint: "main" },
  });

  const computeBindGroups = [0, 1].map((i) =>
    device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: simParamsBuffer } },
        { binding: 1, resource: { buffer: particleBuffers[i] } },
        { binding: 2, resource: { buffer: particleBuffers[1 - i] } },
      ],
    })
  );

  const renderModule = device.createShaderModule({ code: RENDER_WGSL });
  const cloudSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "repeat",
    addressModeV: "repeat",
    addressModeW: "repeat",
  });

  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] });

  const skyPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: renderModule, entryPoint: "vsSky" },
    fragment: { module: renderModule, entryPoint: "fsSky", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const particlePipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: renderModule, entryPoint: "vsParticle" },
    fragment: {
      module: renderModule,
      entryPoint: "fsParticle",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "one", dstFactor: "one", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });

  const renderBindGroups = [0, 1].map((i) =>
    device.createBindGroup({
      layout: renderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: simParamsBuffer } },
        { binding: 1, resource: cloudTexture.createView() },
        { binding: 2, resource: cloudSampler },
        { binding: 3, resource: { buffer: particleBuffers[i] } },
      ],
    })
  );

  const sunSlider = document.getElementById("sun") as HTMLInputElement;
  const cloudSlider = document.getElementById("cloud") as HTMLInputElement;
  const curlSlider = document.getElementById("curl") as HTMLInputElement;

  let mouseActive = 0;
  let mouseX = 0;
  let mouseY = 0;

  canvas.addEventListener("pointermove", (e) => {
    mouseActive = 1;
    const nx = (e.clientX / window.innerWidth) * 2 - 1;
    const ny = -(e.clientY / window.innerHeight) * 2 + 1;
    const aspect = canvas.width / canvas.height;
    mouseX = nx * aspect * 6.0;
    mouseY = ny * 6.0;
  });
  canvas.addEventListener("pointerleave", () => { mouseActive = 0; });

  let frame = 0;
  let lastTime = performance.now();

  function tick(now: number) {
    const dt = Math.min((now - lastTime) * 0.001, 0.033);
    lastTime = now;

    const sunElev = parseFloat(sunSlider.value);
    const cloudCov = parseFloat(cloudSlider.value);
    const curlStr = parseFloat(curlSlider.value);

    document.getElementById("sunVal")!.textContent = sunElev.toFixed(2);
    document.getElementById("cloudVal")!.textContent = cloudCov.toFixed(2);
    document.getElementById("curlVal")!.textContent = curlStr.toFixed(2);

    simParamsData[0] = now * 0.001;
    simParamsData[1] = dt;
    simParamsData[2] = sunElev;
    simParamsData[3] = cloudCov;

    simParamsData[4] = curlStr;
    simParamsData[5] = mouseActive;
    simParamsData[6] = canvas.width / canvas.height;
    simParamsData[7] = 0;

    simParamsData[8] = mouseX;
    simParamsData[9] = mouseY;
    simParamsData[10] = 0.0;
    simParamsData[11] = 0;

    simParamsData[12] = 0.45;
    simParamsData[13] = 0.08;
    simParamsData[14] = 0.12;
    simParamsData[15] = 0;

    device.queue.writeBuffer(simParamsBuffer, 0, simParamsData);

    const ping = frame % 2;
    const pong = 1 - ping;

    const encoder = device.createCommandEncoder();

    const cPass = encoder.beginComputePass();
    cPass.setPipeline(computePipeline);
    cPass.setBindGroup(0, computeBindGroups[ping]);
    cPass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 256));
    cPass.end();

    const rPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    rPass.setBindGroup(0, renderBindGroups[pong]);

    rPass.setPipeline(skyPipeline);
    rPass.draw(3);

    rPass.setPipeline(particlePipeline);
    rPass.draw(6, PARTICLE_COUNT);
    rPass.end();

    device.queue.submit([encoder.finish()]);

    frame++;
    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

init().catch(console.error);
