# unpopped-harness

External test and profile harnesses for [Unpopped](https://github.com/EvansLaboratories/Unpopped)
emitters. Licensed `MIT OR Apache-2.0`.

The contract: a reference kernel is the spec; a candidate is accepted only if it meets its
tier (`Exact` or `Ulp(n)`) against the reference on **every** f32 input (2^32, exhaustive).
A never-wrote check (two different sentinels) catches kernels that write nothing.

Targets today: cpu-c (reference, via MSVC `cl`), CUDA (via `baracuda-cuda-emit` + NVRTC) and
Vulkan (`unpopped-slang` -> `slangc` -> SPIR-V -> `vulkane`).
Depends on published crates only (no path or git deps). Design: `docs/harness-family.md`;
target survey: `docs/SURVEY.md`.

## Running

Windows with Visual Studio (the references are built with `cl`; set `UNPOPPED_VCVARS` to your
`vcvars64.bat` if it is not VS 18 Community). CUDA tests also need an NVIDIA GPU and `nvrtc`; Vulkan tests need `slangc` (on PATH or in
`%VULKAN_SDK%\Bin`) and a Vulkan driver.

    cargo test --release --lib --test planted_bugs --test bench_unary   # needs cl only
    cargo test --release --test gpu_target                                                       # needs a GPU

## CI

`ci.yml` runs fmt, clippy (`--all-targets`, which compiles the GPU tests too), docs and the
cl-only tests (`rewrite_candidates`, ~72 min, runs weekly and on dispatch) on `windows-latest`, plus `cargo deny` and an SPDX header gate. The GPU tests
cannot run on GitHub-hosted runners; they wait on a self-hosted runner and are not `#[ignore]`d.

## Built on

This harness only wraps other people's work, and says so:

- [Unpopped](https://github.com/EvansLaboratories/Unpopped): `unpopped`, `unpopped-vocab` (IR,
  plan, structure keys) and `unpopped-cpu-c` (the portable-C99 reference emitter it compares against).
- [Slang](https://github.com/shader-slang/slang): `slangc` compiles the emitted kernels to SPIR-V.
- [Vulkane](https://github.com/ciresnave/vulkane): the safe Vulkan wrapper the Vulkan target runs on.
- [Baracuda](https://github.com/ciresnave/baracuda): `baracuda-cuda-emit` (the CUDA
  emitter), `baracuda-nvrtc` and `baracuda-driver` (compile and launch).
