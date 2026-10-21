// SPDX-License-Identifier: MIT OR Apache-2.0
//! The harness must be able to fail. Each test plants a known defect and asserts
//! the harness rejects it; the positive control asserts it can also accept.

use std::path::PathBuf;
use unpopped::ir::{input, konst, BinaryOp, Expr, OpDef, UnaryOp};
use unpopped_harness::{compare_unary_f32, Kernel, Report, Tier};
use unpopped_vocab::ElementKind;

fn work() -> PathBuf {
    std::env::temp_dir().join("unpopped-harness")
}

fn op(name: &str, body: Expr) -> Kernel {
    Kernel::from_unary_f32(&OpDef::elementwise(name, 1, &[ElementKind::F32], body))
}

fn run(tag: &str, r: &Kernel, c: &Kernel, tier: Tier, stride: u32) -> Report {
    compare_unary_f32(&work(), tag, r, c, tier, stride).unwrap_or_else(|e| panic!("{tag}: {e}"))
}

fn relu() -> Expr {
    input(0).unary(UnaryOp::Relu)
}

#[test]
fn control_identical_kernels_are_accepted() {
    let r = op("ref_k", relu());
    let c = op("cand_k", relu());
    let rep = run("ctl_same", &r, &c, Tier::Exact, 1);
    assert!(rep.accepted(), "{rep:?}");
    assert_eq!(rep.checked, 1 << 32);
}

fn raw(name: &str, body: &str) -> Kernel {
    Kernel {
        name: name.to_string(),
        source: format!(
            "#include <math.h>
void {name}(const float* in0, float* out, long long n)
{{
{body}
}}
"
        ),
    }
}

fn assert_rejected(rep: &Report) {
    assert!(!rep.accepted(), "harness accepted a planted bug: {rep:?}");
}

/// `x > 0 ? x : 0` loses relu's NaN propagation and its -0.0 preservation.
#[test]
fn rejects_select_formulation_of_relu() {
    let r = op("ref_k", relu());
    let c = op(
        "cand_k",
        input(0)
            .binary(BinaryOp::CmpGt, konst(0.0))
            .select(input(0), konst(0.0)),
    );
    let rep = run("bug_select", &r, &c, Tier::Exact, 1);
    assert_rejected(&rep);
    assert!(rep.mismatches > 0, "{rep:?}");
    assert_eq!(
        rep.cand_never_wrote, 0,
        "rejected for the wrong reason: {rep:?}"
    );
}

/// (x + |x|) * 0.5: wrong on -0.0 and overflows for x > FLT_MAX / 2.
#[test]
fn rejects_abs_formulation_even_with_ulp_slack() {
    let r = op("ref_k", relu());
    let c = op(
        "cand_k",
        (input(0) + input(0).unary(UnaryOp::Abs)) * konst(0.5),
    );
    let rep = run("bug_abs", &r, &c, Tier::Ulp(4), 1);
    assert_rejected(&rep);
}

/// A defect on exactly one input: relu(x) except 12345.0 -> 0.
const RARE: f32 = 12345.0;

fn rare_bug() -> Kernel {
    op(
        "cand_k",
        input(0)
            .binary(BinaryOp::CmpEq, konst(RARE as f64))
            .select(konst(0.0), relu()),
    )
}

#[test]
fn exhaustive_run_finds_a_single_bad_input() {
    let r = op("ref_k", relu());
    let rep = run("bug_rare", &r, &rare_bug(), Tier::Exact, 1);
    assert_rejected(&rep);
    assert_eq!(rep.mismatches, 1, "{rep:?}");
    assert_eq!(rep.first[0].0, RARE.to_bits());
}

/// The same bug is invisible to a strided screen, and a screen can never accept.
#[test]
fn a_strided_screen_misses_it_and_still_cannot_accept() {
    let stride = 4099; // prime; RARE's bit pattern is not a multiple of it
    assert_ne!(RARE.to_bits() % stride, 0);
    let r = op("ref_k", relu());
    let rep = run("bug_rare_screen", &r, &rare_bug(), Tier::Exact, stride);
    assert_eq!(rep.mismatches, 0, "screen unexpectedly saw it: {rep:?}");
    assert!(!rep.accepted(), "a stride>1 run must never accept");
}

/// Tier boundaries: tanh scaled by 1+2^-22 is ~2 ULP off.
#[test]
fn tier_boundaries_are_respected() {
    let th = || input(0).unary(UnaryOp::Tanh);
    let r = op("ref_k", th());
    let c = op("cand_k", th() * konst(1.0 + 2f64.powi(-22)));
    let exact = run("tier_exact", &r, &c, Tier::Exact, 1);
    assert_rejected(&exact);
    let tight = run("tier_u0", &r, &c, Tier::Ulp(0), 1);
    assert_rejected(&tight);
    let max = tight.max_ulp;
    assert!(
        (1..=8).contains(&max),
        "max_ulp {max} not the few-ULP error planted"
    );
    let ok = run("tier_ok", &r, &c, Tier::Ulp(max as u32), 1);
    assert!(ok.accepted(), "{ok:?}");
    let short = run("tier_short", &r, &c, Tier::Ulp(max as u32 - 1), 1);
    assert_rejected(&short);
}

/// The alpha.78 class: launches fine, never writes.
#[test]
fn rejects_a_kernel_that_never_writes() {
    let r = op("ref_k", relu());
    let c = raw("cand_k", "(void)in0; (void)out; (void)n;");
    let rep = run("bug_nowrite", &r, &c, Tier::Exact, 1);
    assert_rejected(&rep);
    assert!(rep.cand_never_wrote > 0, "{rep:?}");
    assert_eq!(rep.ref_never_wrote, 0, "{rep:?}");
}

/// Off-by-one tail: skips the last element of every call.
#[test]
fn rejects_a_kernel_that_skips_the_tail() {
    let r = op("ref_k", relu());
    let c = raw(
        "cand_k",
        "for (long long i = 0; i + 1 < n; ++i) out[i] = (in0[i] < 0.0f ? 0.0f : in0[i]);",
    );
    let rep = run("bug_tail", &r, &c, Tier::Exact, 1);
    assert_rejected(&rep);
    assert!(rep.cand_never_wrote > 0, "{rep:?}");
}
