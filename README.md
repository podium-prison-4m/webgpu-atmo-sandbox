# WebGPU Atmosphere & 250k Particle Sandbox

A zero-dependency, real-time **WebGPU / WGSL** atmospheric scattering, volumetric cloud raymarching, and 250,000-particle physics sandbox running 100% on the client GPU.

Built with raw **TypeScript + Vite + `@webgpu/types`** and hand-written **WGSL** compute/render shaders—no heavy 3D engine wrappers—giving explicit control over bind group layouts, 16-byte uniform buffer alignment, and compute pass scheduling.

## Features

- **Physically Inspired Sky & Aerial Perspective**: Real-time Rayleigh & Mie scattering approximation with smooth transitions from noon sky to golden-hour sunset and nightfall, paired with ACES filmic tone mapping.
- **Volumetric Cloud Raymarching**:
  - One-shot startup compute pass pre-baking a **128×128×128 RGBA8** tileable 3D Worley-Perlin noise texture for hardware trilinear lookups.
  - **Wrenninge's multi-octave scattering approximation** (3 decaying octaves) combined with the **Beer-Powder** term for internal cloud diffusion and silver-lining highlights without multi-bounce path tracing.
  - Adaptive early-ray termination (`transmittance < 0.01`) to keep integrated GPUs (e.g., Apple Silicon MacBooks) cool at 60 FPS.
- **250,000-Particle GPU Compute Simulation**:
  - Lock-free **Ping-Pong double storage buffers** updated at 60 Hz via WGSL compute shaders.
  - Divergence-free **3D Curl Noise** advection combined with thermal buoyancy and directional wind fields.
  - Interactive **3D Vortex Wind Brush**: Hover or drag across the viewport to inject tangential + radial swirl forces into the particle field in real time.

## Tech Stack

- **Language & Build**: TypeScript (strict mode), Vite
- **Graphics API**: Native WebGPU (`@webgpu/types`) & raw WGSL
- **Deployment**: 100% static client-side bundle (zero backend cost; ready for Cloudflare Pages / GitHub Pages)

## Quick Start

### Prerequisites

- Node.js 18+
- A WebGPU-enabled browser (Chrome 113+, Edge 113+, or Safari Technology Preview / macOS Sonoma+)

### Run Locally

```bash
git clone https://github.com/podium-prison-4m/webgpu-atmo-sandbox.git
cd webgpu-atmo-sandbox
npm install
npm run dev
```
