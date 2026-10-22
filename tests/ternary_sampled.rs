// SPDX-License-Identifier: MIT OR Apache-2.0
//! The pairwise-stratified ternary harness must be able to fail, and must never accept.
//! Each planted defect lives in one class pair of two operands (or one edge triple); the sample
//! has to find it.

use std::path::PathBuf;
use unpopped::exactness::ExactnessVerdict;
use unpopped::ir::{input, OpDef};
use unpopped_harness::exactness::verdict_of;
use unpopped_harness::ternary::{
    compare_ternary_f32_sampled, SampleSpec, SampledReport, CELLS, EDGE_TRIPLES, OPERAND_PAIRS,
};
use unpopped_harness::{Kernel, Tier};
use unpopped_vocab::ElementKind;

fn work() -> PathBuf {
    std::env::temp_dir().join("unpopped-harness")
}

/// `select(in0, in1, in2)`: the reference, lowered through cpu-c.
fn select(name: &str) -> Kernel {
    Kernel::from_ternary_f32(&OpDef::elementwise(
        name,
        3,
        &[ElementKind::F32],
        input(0).select(input(1), input(2)),
    ))
}

/// Hand-written select, except that `defect` (a C condition on the operand bits `ab`, `bb`, `cb`)
/// forces 0.
fn defective(name: &str, defect: &str) -> Kernel {
    Kernel {
        name: name.to_string(),
        source: format!(
            "#include <stdint.h>
#include <string.h>
void {name}(const float* in0, const float* in1, const float* in2, float* out, long long n)
{{
    for (long long i = 0; i < n; ++i) {{
        float a = in0[i], b = in1[i], c = in2[i];
        uint32_t ab, bb, cb; memcpy(&ab, &a, 4); memcpy(&bb, &b, 4); memcpy(&cb, &c, 4);
        out[i] = ({defect}) ? 0.0f : (a != 0.0f ? b : c);
    }}
}}
"
        ),
    }
}

fn run(tag: &str, cand: &Kernel) -> SampledReport {
    compare_ternary_f32_sampled(
        &work(),
        tag,
        &select("tern_ref"),
        cand,
        Tier::Exact,
        SampleSpec::default(),
    )
    .unwrap_or_else(|e| panic!("{tag}: {e}"))
}

#[test]
fn control_identical_kernels_conclude_nothing() {
    let s = run("tern_same", &select("tern_cand"));
    assert_eq!(s.report.mismatches, 0, "{s:?}");
    assert!(!s.report.accepted(), "a sample accepted: {s:?}");
    assert_eq!(verdict_of(&s.report), None);
    assert_eq!(s.cells, OPERAND_PAIRS * CELLS);
    assert_eq!(s.edge_triples, EDGE_TRIPLES);
    // By construction, not by chance: every class pair of every operand pair is present in the
    // generated bits. The driver also refuses a run that falls short, so a mutation that leaves
    // a pair to a random class makes `run` fail.
    assert_eq!(s.pair_cells, [CELLS; 3]);
    assert_eq!(s.report.checked, s.spec.triples());
}

/// A defect confined to one class of in1 and one class of in2, with in0 anything. Only the
/// (1,2) operand-pair strata, and the random third class, can reach it.
#[test]
fn catches_a_defect_confined_to_one_class_pair_of_the_last_two_operands() {
    let c = defective(
        "tern_cand_cell",
        "((bb >> 23) == 100u) && ((cb >> 23) == (256u + 200u))",
    );
    let s = run("tern_cell", &c);
    assert!(s.report.mismatches > 0, "missed a one-cell defect: {s:?}");
    assert!(!s.report.accepted());
    assert_eq!(verdict_of(&s.report), Some(ExactnessVerdict::Flushes));
    let (_, b, c) = s.first_operands[0];
    assert_eq!(b >> 23, 100);
    assert_eq!(c >> 23, 256 + 200);
}

/// The same shape on operands 0 and 2: guards the pair indexing in the driver.
#[test]
fn catches_a_defect_confined_to_one_class_pair_of_the_outer_operands() {
    let c = defective(
        "tern_cand_outer",
        "((ab >> 23) == 90u) && ((cb >> 23) == (256u + 30u))",
    );
    let s = run("tern_outer", &c);
    assert!(s.report.mismatches > 0, "{s:?}");
    let (a, _, c) = s.first_operands[0];
    assert_eq!(a >> 23, 90);
    assert_eq!(c >> 23, 256 + 30);
}

/// Exact edge values a random mantissa rarely produces: the edge grid finds it.
#[test]
fn catches_a_defect_on_one_edge_triple() {
    let c = defective(
        "tern_cand_edge",
        "ab == 0x7f800000u && bb == 0x80000000u && cb == 0x7fa00001u",
    );
    let s = run("tern_edge", &c);
    assert!(
        s.report.mismatches > 0,
        "missed an edge-triple defect: {s:?}"
    );
    assert_eq!(verdict_of(&s.report), Some(ExactnessVerdict::Flushes));
}
