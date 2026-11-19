# webgpu-atmo-sandbox

A real-time WebGPU sandbox for **volumetric clouds** and **GPU spatial hashing particles**.

Weekend project — experimenting with what's possible when the whole pipeline (simulation + rendering) lives on the GPU.

## What's in the box

- **Volumetric clouds** — raymarched cloud layer driven by procedural noise (Worley / FBM), with density, coverage and lighting controls
- **Particle system** — GPU compute particles sorted into a spatial hash grid for fast neighbor lookups and interaction
- **Compute-first** — simulation in WGSL compute shaders, minimal CPU work per frame

## Stack

- TypeScript
- WGSL (WebGPU Shading Language)
- WebGPU (`navigator.gpu`) — no WebGL fallback planned

## Status

🚧 Early scaffolding. Boilerplate (TS setup, WGSL shader plumbing, buffer bindings) is being wired up first; the interesting shaders come after.

## Getting started

```bash
npm install
npm run dev
```

Then open the dev server URL in a WebGPU-capable browser (Chrome / Edge 113+).

## License

MIT
