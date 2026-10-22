# Changelog

## 0.0.4

Docs only: the "cpu-c `Max` vs IR docs" open fact is resolved (Unpopped 0.20.15 corrected the IR comment).

## 0.0.3

`Options::slangc_args` and `vulkan::DENORM_PRESERVE` (`-denorm-mode-fp32 preserve`). With it the
Slang relu is exact against cpu-c over all 2^32 f32 on both AMD Radeon 610M and RTX 4070 Laptop,
`waived == 0` (the 0.0.2 finding was the device flushing negative subnormals by default).
`declares_denorm_preserve` reads the SPIR-V capability word, and a test uses it so a flag that
`slangc` ignored cannot pass for one that took effect. NVIDIA reports
`shaderDenormPreserveFloat32 = false` yet honours the mode, so the property is not a usable gate;
this measurement is.

## 0.0.2

Third target: Vulkan. `vulkan::compare_unary_f32_vulkan` runs an `unpopped-slang` candidate
(`slangc` -> SPIR-V -> `vulkane` 0.16.0) on any Vulkan compute device against the cpu-c
reference, over all 2^32 f32, through the same loop as CUDA (`gpu::compare_with_runner`, factored
out of the CUDA path with no behaviour change). `Report::waived` counts mismatches the caller
declared known (`Options::waive`), kept out of `mismatches` so a test can pin the exact size.

Finding, pinned by `tests/vulkan_target.rs`: on both AMD Radeon 610M and RTX 4070 Laptop the
Slang relu differs from cpu-c/CUDA on exactly the 8,388,607 negative subnormals (the device
flushes them in `x < 0.0`, then returns `x` unchanged). Waived and counted; a second test shows
the waiver does not hide a real bug (`abs` is still rejected).

## 0.0.1

Fix: the first CI run on `windows-latest` failed (`'cl' is not recognized`) because the VS 18
`vcvars64.bat` path was hard-coded and the `UNPOPPED_VCVARS` override named a VS 2022 path that
is not what the runner has. `vcvars()` now asks `vswhere` for the newest install with the C++
tools (override and VS 18 default kept), and `ci.yml` no longer sets the variable.

## 0.0.0

Initial import of the Unpopped test harness from the scratch crate `unpopped-evolve-scratch`:
exhaustive 2^32 f32 kernel comparison (tiers Exact / Ulp(n), never-wrote sentinels), cpu-c and
CUDA targets, per-op and chain timings, planted-bug tests. CI, cargo-deny and an SPDX gate from
day one.
