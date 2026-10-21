// SPDX-License-Identifier: MIT OR Apache-2.0
//! The CUDA target through the same contract: exhaustive f32 compare, GPU candidate
//! against the cpu-c reference, plus planted bugs the GPU path must reject.

use std::path::PathBuf;
use unpopped::ir::{input, konst, BinaryOp, Expr, OpDef, UnaryOp};
use unpopped_harness::gpu::compare_unary_f32_gpu;
use unpopped_harness::{Kernel, Tier};
use unpopped_vocab::ElementKind;

fn work() -> PathBuf {
    std::env::temp_dir().join("unpopped-harness-gpu")
}

fn def(name: &str, body: Expr) -> OpDef {
    OpDef::elementwise(name, 1, &[ElementKind::F32], body)
}

fn relu() -> Expr {
    input(0).unary(UnaryOp::Relu)
}

#[test]
fn gpu_relu_equals_cpu_relu_exhaustively() {
    let r = Kernel::from_unary_f32(&def("ref_k", relu()));
    let c = Kernel::from_unary_f32_cuda(&def("cand_k", relu()));
    let rep = compare_unary_f32_gpu(&work(), "gpu_relu", &r, &c, Tier::Exact, 1).unwrap();
    println!("{rep:?}");
    assert!(rep.accepted(), "{rep:?}");
}

/// Control: a planted select-formulation of relu must be rejected on the GPU path too.
#[test]
fn gpu_rejects_planted_select_relu() {
    let r = Kernel::from_unary_f32(&def("ref_k", relu()));
    let bad = input(0)
        .binary(BinaryOp::CmpGt, konst(0.0))
        .select(input(0), konst(0.0));
    let c = Kernel::from_unary_f32_cuda(&def("cand_k", bad));
    let rep = compare_unary_f32_gpu(&work(), "gpu_bad", &r, &c, Tier::Exact, 1).unwrap();
    assert!(!rep.accepted() && rep.mismatches > 0, "{rep:?}");
}

/// Control: a kernel that never writes is caught by the sentinel test.
#[test]
fn gpu_rejects_kernel_that_never_writes() {
    let r = Kernel::from_unary_f32(&def("ref_k", relu()));
    let c = Kernel {
        name: "cand_k".into(),
        source:
            "extern \"C\" __global__ void cand_k(const float* in0, float* out, long long n) {}\n"
                .into(),
    };
    let rep = compare_unary_f32_gpu(&work(), "gpu_nw", &r, &c, Tier::Exact, 65537).unwrap();
    assert!(rep.cand_never_wrote > 0 && !rep.accepted(), "{rep:?}");
}

fn max0() -> Expr {
    input(0).binary(BinaryOp::Max, konst(0.0))
}

/// Open fact from #83: is `Max(x, 0)` bit-identical to Relu on CUDA too?
#[test]
fn gpu_max_zero_equals_cpu_relu_exhaustively() {
    let r = Kernel::from_unary_f32(&def("ref_k", relu()));
    let c = Kernel::from_unary_f32_cuda(&def("cand_k", max0()));
    let rep = compare_unary_f32_gpu(&work(), "gpu_max0", &r, &c, Tier::Exact, 1).unwrap();
    println!("MAX0 {rep:?}");
    assert!(rep.accepted(), "{rep:?}");
}

/// Control: IEEE fmax(x, 0) scrubs NaN, so it must differ from Relu on the same path.
#[test]
fn gpu_fmax_ieee_zero_differs_from_cpu_relu() {
    let r = Kernel::from_unary_f32(&def("ref_k", relu()));
    let c = Kernel::from_unary_f32_cuda(&def(
        "cand_k",
        input(0).binary(BinaryOp::FmaxIeee, konst(0.0)),
    ));
    let rep = compare_unary_f32_gpu(&work(), "gpu_fmax0", &r, &c, Tier::Exact, 1).unwrap();
    println!("FMAX0 {rep:?}");
    assert!(!rep.accepted() && rep.mismatches > 0, "{rep:?}");
}
