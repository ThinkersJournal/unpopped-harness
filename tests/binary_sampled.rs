// SPDX-License-Identifier: MIT OR Apache-2.0
//! The stratified binary harness must be able to fail, and must never accept.
//! Each planted defect flushes on ONE class of operand pair; the sample has to find it.

use std::path::PathBuf;
use unpopped::exactness::ExactnessVerdict;
use unpopped::ir::{input, OpDef};
use unpopped_harness::binary::{compare_binary_f32_sampled, SampleSpec, SampledReport, CELLS};
use unpopped_harness::exactness::verdict_of;
use unpopped_harness::{Kernel, Tier};
use unpopped_vocab::ElementKind;

fn work() -> PathBuf {
    std::env::temp_dir().join("unpopped-harness")
}

fn affine(name: &str) -> Kernel {
    Kernel::from_binary_f32(&OpDef::elementwise(
        name,
        2,
        &[ElementKind::F32],
        input(0) * input(1) + input(0),
    ))
}

/// `a*b+a`, except that `defect` (a C condition on `a`, `b` and their bits `ab`, `bb`) forces 0.
fn defective(name: &str, defect: &str) -> Kernel {
    Kernel {
        name: name.to_string(),
        source: format!(
            "#include <math.h>
#include <stdint.h>
#include <string.h>
void {name}(const float* in0, const float* in1, float* out, long long n)
{{
    for (long long i = 0; i < n; ++i) {{
        float a = in0[i], b = in1[i];
        uint32_t ab, bb; memcpy(&ab, &a, 4); memcpy(&bb, &b, 4);
        out[i] = ({defect}) ? 0.0f : ((a * b) + a);
    }}
}}
"
        ),
    }
}

fn run(tag: &str, cand: &Kernel) -> SampledReport {
    compare_binary_f32_sampled(
        &work(),
        tag,
        &affine("bin_ref"),
        cand,
        Tier::Exact,
        SampleSpec::default(),
    )
    .unwrap_or_else(|e| panic!("{tag}: {e}"))
}

#[test]
fn control_identical_kernels_conclude_nothing() {
    let s = run("bin_same", &affine("bin_cand"));
    assert_eq!(s.report.mismatches, 0, "{s:?}");
    assert!(!s.report.accepted(), "a sample accepted: {s:?}");
    assert_eq!(verdict_of(&s.report), None);
    // The verdict can say what it covered.
    assert_eq!(s.cells, CELLS);
    assert_eq!(s.edge_pairs, 256);
    assert_eq!(s.report.checked, s.spec.pairs());
}

/// Positive a in one binade, b negative in another: no edge value is in either class, so only
/// the stratified cells can find it.
#[test]
fn catches_a_defect_confined_to_one_operand_class_pair() {
    let c = defective(
        "bin_cand_cell",
        "((ab >> 23) == 100u) && ((bb >> 23) == (256u + 200u))",
    );
    let s = run("bin_cell", &c);
    assert!(s.report.mismatches > 0, "missed a one-cell defect: {s:?}");
    assert!(
        s.report.mismatches <= u64::from(s.spec.per_cell),
        "defect lives in one cell, saw {}",
        s.report.mismatches
    );
    assert!(!s.report.accepted());
    assert_eq!(verdict_of(&s.report), Some(ExactnessVerdict::Flushes));
    let (a, b) = s.first_operands[0];
    assert_eq!(a >> 23, 100);
    assert_eq!(b >> 23, 256 + 200);
}

/// A defect on exact values (+inf with -0) that a random mantissa rarely produces: the edge grid
/// finds it.
#[test]
fn catches_a_defect_on_one_edge_pair() {
    let c = defective("bin_cand_edge", "ab == 0x7f800000u && bb == 0x80000000u");
    let s = run("bin_edge", &c);
    assert!(s.report.mismatches > 0, "missed an edge-pair defect: {s:?}");
    assert_eq!(verdict_of(&s.report), Some(ExactnessVerdict::Flushes));
}

/// Subnormal-flush shape, the case that motivated the module: flushes a subnormal result only
/// when both operands are subnormal.
#[test]
fn catches_a_flush_of_subnormal_pairs() {
    let c = defective(
        "bin_cand_sub",
        "((ab & 0x7f800000u) == 0u) && ((bb & 0x7f800000u) == 0u) && (ab & 0x7fffffu)",
    );
    let s = run("bin_sub", &c);
    assert!(s.report.mismatches > 0, "{s:?}");
    assert_eq!(verdict_of(&s.report), Some(ExactnessVerdict::Flushes));
}
