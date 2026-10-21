// SPDX-License-Identifier: MIT OR Apache-2.0
//! Track 3: candidate algebraic rewrites for unary f32 bodies, each checked
//! exhaustively (all 2^32 inputs) against the original through cpu-c.
//! A rewrite is only a candidate for unpopped's e-graph if it is Exact; the Ulp
//! rows measure how far the near-misses are. Never asserts equivalence: it
//! prints the verdict per candidate, and asserts only that the run completed.

use std::path::PathBuf;
use unpopped::ir::{input, konst, BinaryOp, Expr, OpDef, UnaryOp};
use unpopped_harness::{compare_unary_f32, Kernel, Tier};
use unpopped_vocab::ElementKind;

fn k(name: &str, body: Expr) -> Kernel {
    Kernel::from_unary_f32(&OpDef::elementwise(name, 1, &[ElementKind::F32], body))
}
fn x() -> Expr {
    input(0)
}
fn u(op: UnaryOp) -> Expr {
    x().unary(op)
}

#[test]
fn rewrite_candidates() {
    let w: PathBuf = std::env::temp_dir().join("unpopped-harness-t3");
    let one = || konst(1.0);
    let c: Vec<(&str, Expr, Expr)> = vec![
        ("sqr = x*x", u(UnaryOp::Sqr), x() * x()),
        ("recip = 1/x", u(UnaryOp::Recip), one() / x()),
        (
            "rsqrt = 1/sqrt",
            u(UnaryOp::Rsqrt),
            one() / u(UnaryOp::Sqrt),
        ),
        (
            "sigmoid = 1/(1+exp(-x))",
            u(UnaryOp::Sigmoid),
            one() / (one() + u(UnaryOp::Neg).unary(UnaryOp::Exp)),
        ),
        (
            "silu = x*sigmoid",
            u(UnaryOp::Silu),
            x() * u(UnaryOp::Sigmoid),
        ),
        (
            "silu = x/(1+exp(-x))",
            u(UnaryOp::Silu),
            x() / (one() + u(UnaryOp::Neg).unary(UnaryOp::Exp)),
        ),
        (
            "sqrt(sqr x) = abs",
            u(UnaryOp::Sqr).unary(UnaryOp::Sqrt),
            u(UnaryOp::Abs),
        ),
        (
            "abs*abs = sqr",
            u(UnaryOp::Abs) * u(UnaryOp::Abs),
            u(UnaryOp::Sqr),
        ),
        ("neg(neg x) via 0-x", u(UnaryOp::Neg), konst(0.0) - x()),
        ("neg = x*-1", u(UnaryOp::Neg), x() * konst(-1.0)),
        ("x+x = x*2", x() + x(), x() * konst(2.0)),
        (
            "relu = max(x,0)",
            u(UnaryOp::Relu),
            x().binary(BinaryOp::Max, konst(0.0)),
        ),
        (
            "exp2 = exp(x*ln2)",
            u(UnaryOp::Exp2),
            (x() * konst(std::f64::consts::LN_2)).unary(UnaryOp::Exp),
        ),
        (
            "tanh = 2*sigmoid(2x)-1",
            u(UnaryOp::Tanh),
            konst(2.0) * (x() * konst(2.0)).unary(UnaryOp::Sigmoid) - one(),
        ),
    ];
    for (i, (name, lhs, rhs)) in c.into_iter().enumerate() {
        let r = k(&format!("ref_{i}"), lhs);
        let cd = k(&format!("cand_{i}"), rhs);
        for tier in [Tier::Exact, Tier::Ulp(4)] {
            match compare_unary_f32(&w, &format!("t3_{i}"), &r, &cd, tier, 1) {
                Ok(rep) => {
                    println!(
                        "{name:28} {tier:?}: accepted={} mism={} max_ulp={} first={:?}",
                        rep.accepted(),
                        rep.mismatches,
                        rep.max_ulp,
                        rep.first.first()
                    );
                    if rep.accepted() {
                        break;
                    }
                }
                Err(e) => {
                    println!("{name:28} ERROR {e}");
                    break;
                }
            }
        }
    }
}
