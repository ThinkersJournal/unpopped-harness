// SPDX-License-Identifier: MIT OR Apache-2.0
//! The f64 screens (unary, binary, ternary) must be able to fail, and must never accept.
//! Each planted defect lives in one class cell (or one edge tuple); the sample has to find it.

use std::path::PathBuf;
use unpopped::exactness::ExactnessVerdict;
use unpopped::ir::{input, OpDef};
use unpopped_harness::double::{
    compare_binary_f64_sampled, compare_ternary_f64_sampled, compare_unary_f64_sampled, SampleSpec,
    SampledReport, CLASS_PAIRS, OPERAND_CLASSES,
};
use unpopped_harness::exactness::verdict_of;
use unpopped_harness::{Kernel, Tier};
use unpopped_vocab::ElementKind;

fn work() -> PathBuf {
    std::env::temp_dir().join("unpopped-harness")
}

fn op(name: &str, arity: u8, body: unpopped::ir::Expr) -> OpDef {
    OpDef::elementwise(name, arity, &[ElementKind::F64], body)
}

fn unary(name: &str) -> Kernel {
    Kernel::from_unary_f64(&op(name, 1, input(0) * input(0) + input(0)))
}
fn binary(name: &str) -> Kernel {
    Kernel::from_binary_f64(&op(name, 2, input(0) * input(1) + input(0)))
}
fn ternary(name: &str) -> Kernel {
    Kernel::from_ternary_f64(&op(name, 3, input(0).select(input(1), input(2))))
}

const PRELUDE: &str = "#include <stdint.h>\n#include <string.h>\n";

/// Hand-written candidates equal to the references except that `defect` (a C condition on the
/// operand bits `ab`, `bb`, `cb`) forces 0.
fn defective(name: &str, arity: usize, defect: &str) -> Kernel {
    let (sig, load, calc) = match arity {
        1 => (
            "const double* in0",
            "double a = in0[i], b = 0, c = 0;",
            "(a * a) + a",
        ),
        2 => (
            "const double* in0, const double* in1",
            "double a = in0[i], b = in1[i], c = 0;",
            "(a * b) + a",
        ),
        _ => (
            "const double* in0, const double* in1, const double* in2",
            "double a = in0[i], b = in1[i], c = in2[i];",
            "(a != 0.0 ? b : c)",
        ),
    };
    Kernel {
        name: name.to_string(),
        source: format!(
            "{PRELUDE}void {name}({sig}, double* out, long long n)
{{
    for (long long i = 0; i < n; ++i) {{
        {load}
        uint64_t ab, bb, cb; memcpy(&ab, &a, 8); memcpy(&bb, &b, 8); memcpy(&cb, &c, 8);
        out[i] = ({defect}) ? 0.0 : ({calc});
    }}
}}
"
        ),
    }
}

fn run(arity: usize, tag: &str, cand: &Kernel, tier: Tier) -> SampledReport {
    let spec = SampleSpec::for_arity(arity as u32);
    let r = match arity {
        1 => compare_unary_f64_sampled(&work(), tag, &unary("f64_ref"), cand, tier, spec),
        2 => compare_binary_f64_sampled(&work(), tag, &binary("f64_ref"), cand, tier, spec),
        _ => compare_ternary_f64_sampled(&work(), tag, &ternary("f64_ref"), cand, tier, spec),
    };
    r.unwrap_or_else(|e| panic!("{tag}: {e}"))
}

fn flushes(s: &SampledReport) {
    assert!(s.report.mismatches > 0, "missed the planted defect: {s:?}");
    assert!(!s.report.accepted(), "a sample accepted: {s:?}");
    assert_eq!(verdict_of(&s.report), Some(ExactnessVerdict::Flushes));
}

/// Identical kernels: nothing seen, nothing concluded, and the report says what it covered.
fn control(arity: usize, cand: Kernel) {
    let s = run(arity, &format!("f64_same{arity}"), &cand, Tier::Exact);
    let a = arity as u32;
    assert_eq!(s.report.mismatches, 0, "{s:?}");
    assert!(!s.report.accepted(), "a sample accepted: {s:?}");
    assert_eq!(verdict_of(&s.report), None);
    assert_eq!(s.cells, SampleSpec::cells(a));
    assert_eq!(s.edge_tuples, 16u64.pow(a));
    assert_eq!(s.report.checked, s.spec.tuples(a));
    // Read back from the generated operand bits; the driver refuses a short run.
    let full = if arity == 1 {
        OPERAND_CLASSES
    } else {
        CLASS_PAIRS
    };
    let sets = if arity == 3 { 3 } else { 1 };
    assert_eq!(s.covered, vec![full; sets]);
}

#[test]
fn control_identical_unary_kernels_conclude_nothing() {
    control(1, unary("f64_cand"));
}
#[test]
fn control_identical_binary_kernels_conclude_nothing() {
    control(2, binary("f64_cand"));
}
#[test]
fn control_identical_ternary_kernels_conclude_nothing() {
    control(3, ternary("f64_cand"));
}

/// One binade of a: no edge value is in it, so only the class strata can find it. The exponent
/// (1100) does not fit f32's 8 bits, so a sampler that kept f32's classes would miss it.
#[test]
fn unary_catches_a_defect_confined_to_one_class() {
    let c = defective("f64_cand_u", 1, "(ab >> 52) == 1100u");
    let s = run(1, "f64_u_cell", &c, Tier::Exact);
    flushes(&s);
    assert!(s.report.mismatches <= u64::from(s.spec.per_cell));
    assert_eq!(s.first_failures[0].operands[0] >> 52, 1100);
}

#[test]
fn binary_catches_a_defect_confined_to_one_class_pair() {
    let c = defective(
        "f64_cand_b",
        2,
        "((ab >> 52) == 1100u) && ((bb >> 52) == (2048u + 1500u))",
    );
    let s = run(2, "f64_b_cell", &c, Tier::Exact);
    flushes(&s);
    assert!(s.report.mismatches <= u64::from(s.spec.per_cell));
    let f = &s.first_failures[0];
    assert_eq!(f.operands[0] >> 52, 1100);
    assert_eq!(f.operands[1] >> 52, 2048 + 1500);
}

/// Classes of in1 and in2 only, then of in0 and in2 only: guards the operand-pair indexing.
#[test]
fn ternary_catches_a_defect_confined_to_one_class_pair() {
    let c = defective(
        "f64_cand_tlast",
        3,
        "((bb >> 52) == 1100u) && ((cb >> 52) == (2048u + 1500u))",
    );
    let s = run(3, "f64_t_last", &c, Tier::Exact);
    flushes(&s);
    let f = &s.first_failures[0];
    assert_eq!(f.operands[1] >> 52, 1100);
    assert_eq!(f.operands[2] >> 52, 2048 + 1500);

    let c = defective(
        "f64_cand_touter",
        3,
        "((ab >> 52) == 900u) && ((cb >> 52) == (2048u + 300u))",
    );
    let s = run(3, "f64_t_outer", &c, Tier::Exact);
    flushes(&s);
    let f = &s.first_failures[0];
    assert_eq!(f.operands[0] >> 52, 900);
    assert_eq!(f.operands[2] >> 52, 2048 + 300);
}

/// Exact edge values a random mantissa never produces: the edge grid finds them.
#[test]
fn every_arity_catches_a_defect_on_one_edge_tuple() {
    let inf = "ab == 0x7ff0000000000000ull";
    let c1 = defective("f64_cand_e1", 1, inf);
    flushes(&run(1, "f64_e1", &c1, Tier::Exact));
    let c2 = defective(
        "f64_cand_e2",
        2,
        &format!("{inf} && bb == 0x8000000000000000ull"),
    );
    flushes(&run(2, "f64_e2", &c2, Tier::Exact));
    let c3 = defective(
        "f64_cand_e3",
        3,
        &format!("{inf} && bb == 0x8000000000000000ull && cb == 0x7ff4000000000001ull"),
    );
    flushes(&run(3, "f64_e3", &c3, Tier::Exact));
}

/// The subnormal-flush shape: a candidate that flushes subnormal inputs to zero. In f64 the
/// subnormals are exponent field 0, the class f32's mantissa-width constants would mislabel.
#[test]
fn unary_catches_a_flush_of_subnormals() {
    let c = defective(
        "f64_cand_sub",
        1,
        "((ab & 0x7ff0000000000000ull) == 0ull) && (ab & 0xfffffffffffffull)",
    );
    flushes(&run(1, "f64_sub", &c, Tier::Exact));
}

/// The ULP tier on f64: a candidate one ulp off passes `Ulp(1)` and fails `Exact`, and the
/// measured distance is exactly 1.
#[test]
fn ulp_tier_measures_f64_distance() {
    let off_by_one = Kernel {
        name: "f64_cand_ulp".to_string(),
        source: format!(
            "{PRELUDE}void f64_cand_ulp(const double* in0, double* out, long long n)
{{
    for (long long i = 0; i < n; ++i) {{
        double a = in0[i], r = (a * a) + a;
        uint64_t b; memcpy(&b, &r, 8);
        if (r == r && (b & 0x7ff0000000000000ull) != 0x7ff0000000000000ull
            && (b & 0x7fffffffffffffffull) != 0ull) b += 1;
        memcpy(&r, &b, 8);
        out[i] = r;
    }}
}}
"
        ),
    };
    let loose = run(1, "f64_ulp1", &off_by_one, Tier::Ulp(1));
    assert_eq!(loose.report.mismatches, 0, "{loose:?}");
    assert_eq!(loose.report.max_ulp, 1);
    flushes(&run(1, "f64_ulp0", &off_by_one, Tier::Exact));
}
