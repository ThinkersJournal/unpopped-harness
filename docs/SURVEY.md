# Target dispatch survey (read-only), 2026-10-19

Read at `origin/main` of each repo after `git fetch`: baracuda 12546da5, vulkane 962e5c3,
fuel c0459c26e, aocl 7fbe68f. crates.io versions checked with `cargo search` the same day.
Nothing was built or run for this survey; every row is a reading of source or manifests.

| Target | What it gives a harness | What is missing | Published |
|---|---|---|---|
| CUDA: `baracuda-cuda-emit` | `Cuda` Backend for unpopped (IR -> .cu) and `NvrtcCompiler` (.cu -> PTX, per `TargetId`, feature `nvrtc`). Has a `device-tests` feature, a self-hosted device runner, and `docs/DEVICE-TESTS.md` | Launch: load the PTX and run it (`baracuda-driver`/`baracuda-runtime`, not read yet) | cuda-emit 0.20.1; baracuda 0.0.1-alpha.126 |
| Vulkan: `vulkane` | Safe wrappers: buffers, compute pipelines, command recording, `dispatch`, timestamp `QueryPool` (+ `timestamp_period`). Examples `compute_square`, `fill_buffer`. Takes SPIR-V from any producer | Slang -> SPIR-V is outside it (we have `slangc`); descriptor/pipeline glue is ours to write | vulkane 0.16.0 |
| Fuel Slang corpus | 147 `.slang` + 20 `.glsl` kernels in `fuel-kernels-source`; precompiled SPIR-V in `fuel-vulkan-kernels`; `fuel-vulkan-backend` has dispatch and tests | Fuel's own binding layout, not Unpopped's calling convention; a corpus and hand-written comparator for `unpopped-slang`, not a drop-in runner | 0.20.14 |
| AOCL (`aocl-*`) | BLAS, LAPACK, FFT, math, RNG, sparse wrappers | Library calls, not an elementwise-body executor | not checked |
| oneMKL (`onemkl` 0.2.0) | Safe wrapper over oneMKL | same as AOCL; and this machine is AMD, so oneMKL CPU use needs checking | onemkl 0.2.0 |

## Reading

- AOCL and oneMKL are BLAS/FFT-class libraries. They are oracles and comparison points for
  contraction and transform kernels, not targets for the unary/binary elementwise harness.
  Park them for the contraction phase.
- The two real second-target candidates are CUDA and Vulkan. Hardware is here for both
  (RTX 4070 Laptop with `nvcc` 13.3; AMD 610M iGPU and the 4070 for Vulkan; `slangc` present).
- Recommendation: CUDA first. Unpopped's own CUDA emitter and an NVRTC compile step are
  already published, so the new work is one launch wrapper. Vulkan needs the whole
  slangc -> pipeline -> descriptors -> dispatch path written. CUDA also gives a three-way
  emitter comparison (cpu-c, cuda, slang) as soon as Vulkan follows.

## Verified by the CUDA spike (2026-10-19, `tests/cuda_spike.rs`)

- NVRTC runtime is present: `nvrtc64_130_0.dll` in `CUDA/v13.3/bin/x64`, which is on PATH, and
  the name is in baracuda's candidate list. Nothing was installed.
- `baracuda-cuda-emit 0.20.1` builds with `unpopped 0.20.13` (one unpopped in the tree).
- Launch path works from published crates: Cuda backend -> `baracuda_nvrtc::Program::compile`
  -> `baracuda_driver::{Module, Kernel::launch}` on the RTX 4070 (cc 8.9). Relu over 1,047,816
  inputs (NaN, +-0, +-inf, signalling NaN included): 0 mismatches vs a host relu. Control: with the
  host relu mutated to `x.max(0.0)` the same test fails with 4,097 mismatches (NaN inputs), so
  the comparison can fail.
- Emitted CUDA kernel has the same signature as cpu-c's: `(const float* in0, float* out, long long n)`,
  grid-stride loop, so one harness driver can serve both.
