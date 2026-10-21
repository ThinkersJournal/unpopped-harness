# Changelog

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
