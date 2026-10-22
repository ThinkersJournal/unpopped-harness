# Changelog

## 0.0.7

`ternary::compare_ternary_f32_sampled` and `Kernel::from_ternary_f32`: ternary f32 bodies (Select,
Clamp, Fma) get a screen. 512^3 operand-class cells are too many, so every *pair* of operands is
covered (three operand pairs x 512 x 512 class pairs x `per_cell` triples, the third operand's class
random) plus a 16 x 16 x 16 edge grid. A defect needing all three classes at once is reached only by
chance. A sample never accepts: the verdict is `Flushes` or nothing. The result records cells, edge
triples, seed and sample size.

## 0.0.6

`binary::compare_binary_f32_sampled` and `Kernel::from_binary_f32`: binary f32 bodies get a stratified
screen. Every ordered pair of operand classes (sign x biased exponent, 512 x 512 cells) gets
`per_cell` seeded pairs with corner mantissas, plus a 16 x 16 grid of edge values. The result
records the cells, edge pairs, seed and sample size. A sample never accepts: the verdict is
`Flushes` or nothing (`exactness::verdict_of`). Tests plant defects confined to one operand-class
pair, one edge pair and subnormal pairs; the stratification finds each, and identical kernels
conclude nothing.

## 0.0.5

`exactness::{verdict_of, unary_f32_record}` and `vulkan::device_stamp`: the harness now emits the
`unpopped::exactness` record (Unpopped 0.20.19) for a run. `Exact` only from an accepted
exhaustive run with nothing waived; `Flushes` from any difference, waived or not, or an unwritten
output; a clean sample concludes nothing. The stamp carries the device, API version, vendor and
driver version, and the `slangc` arguments. A test runs the Slang relu on every Vulkan device with
and without `DENORM_PRESERVE` and feeds the records to `precision_of_measured`: the declared
precision follows the measurement on each device. Dependencies move to Unpopped 0.20.19.

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
