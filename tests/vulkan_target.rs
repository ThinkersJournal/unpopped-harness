// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Slang emitter through the same contract: exhaustive f32 compare on every Vulkan device,
//! candidate (`unpopped-slang` -> `slangc` -> SPIR-V -> `vulkane`) against the cpu-c reference,
//! plus planted bugs the Vulkan path must reject. Needs `slangc` and a Vulkan driver.

use std::path::PathBuf;
use unpopped::ir::{input, konst, BinaryOp, Expr, OpDef, UnaryOp};
use unpopped_harness::vulkan::{
    compare_unary_f32_vulkan, declares_denorm_preserve, slang_to_spirv, vulkan_devices, Options,
    DENORM_PRESERVE,
};
use unpopped_harness::{Kernel, Tier};
use unpopped_vocab::ElementKind;

fn work() -> PathBuf {
    std::env::temp_dir().join("unpopped-harness-vulkan")
}

fn def(name: &str, body: Expr) -> OpDef {
    OpDef::elementwise(name, 1, &[ElementKind::F32], body)
}

fn relu() -> Expr {
    input(0).unary(UnaryOp::Relu)
}

/// Negative subnormals (not -0.0). Every Vulkan device tried so far (AMD 610M, RTX 4070 Laptop)
/// flushes them in the compare `x < 0.0` that the Slang relu is spelled with, then returns `x`
/// unchanged, where cpu-c and CUDA return 0. SPIR-V leaves f32 denormal behaviour to the
/// implementation unless `DenormPreserve` is declared, and the Slang emitter declares nothing.
/// Waived and counted, not hidden: see `vulkan_waiver_does_not_cover_a_real_bug`.
fn negative_denormal(bits: u32) -> bool {
    bits & 0x8000_0000 != 0 && bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0
}

/// 2^23 - 1 patterns have a zero exponent and a nonzero mantissa with the sign set.
const NEG_DENORMALS: u64 = (1 << 23) - 1;

/// Stride for the screening runs; the exhaustive test below uses 1.
const SCREEN: u32 = 65_537;

#[test]
fn a_vulkan_device_with_compute_exists() {
    let devs = vulkan_devices().unwrap();
    println!("vulkan compute devices: {devs:?}");
    assert!(!devs.is_empty());
}

#[test]
fn vulkan_relu_equals_cpu_relu_on_every_device_exhaustively() {
    let r = Kernel::from_unary_f32(&def("ref_k", relu()));
    let c = Kernel::from_unary_f32_slang(&def("cand_k", relu()));
    let only: Option<usize> = std::env::var("UNPOPPED_VULKAN_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok());
    let mut failed = Vec::new();
    for (i, name) in vulkan_devices().unwrap().iter().enumerate() {
        if only.is_some_and(|o| o != i) {
            continue;
        }
        let tag = format!("vk_relu_{i}");
        let rep = compare_unary_f32_vulkan(
            &work(),
            &tag,
            i,
            &r,
            &c,
            Options {
                waive: negative_denormal,
                ..Options::exhaustive(Tier::Exact)
            },
        )
        .unwrap();
        println!("{name}: {rep:?}");
        if !rep.accepted() || ![0, NEG_DENORMALS].contains(&rep.waived) {
            failed.push(format!("{name}: {rep:?}"));
        }
    }
    // Every device is run before any is judged, so one failing device cannot hide the others.
    assert!(failed.is_empty(), "{failed:#?}");
}

/// Control: a planted select-formulation of relu must be rejected on the Vulkan path too.
#[test]
fn vulkan_rejects_planted_select_relu() {
    let r = Kernel::from_unary_f32(&def("ref_k", relu()));
    let bad = input(0)
        .binary(BinaryOp::CmpGt, konst(0.0))
        .select(input(0), konst(0.0));
    let c = Kernel::from_unary_f32_slang(&def("cand_k", bad));
    let rep = compare_unary_f32_vulkan(
        &work(),
        "vk_bad",
        0,
        &r,
        &c,
        Options::exhaustive(Tier::Exact),
    )
    .unwrap();
    assert!(!rep.accepted() && rep.mismatches > 0, "{rep:?}");
}

/// Control: a kernel that never writes is caught by the sentinel test.
#[test]
fn vulkan_rejects_kernel_that_never_writes() {
    let r = Kernel::from_unary_f32(&def("ref_k", relu()));
    let c = Kernel {
        name: "cand_k".into(),
        source: "StructuredBuffer<float> input0;\nRWStructuredBuffer<float> output;\n\
                 [numthreads(256, 1, 1)]\nvoid cand_k(uint3 tid : SV_DispatchThreadID) {}\n"
            .into(),
    };
    let rep = compare_unary_f32_vulkan(
        &work(),
        "vk_nw",
        0,
        &r,
        &c,
        Options {
            stride: SCREEN,
            ..Options::exhaustive(Tier::Exact)
        },
    )
    .unwrap();
    assert!(rep.cand_never_wrote > 0 && !rep.accepted(), "{rep:?}");
}

/// Control: the waiver covers only negative subnormals. A candidate that is wrong on ordinary
/// negatives (`abs`, not relu) is still rejected with the same waiver in force.
#[test]
fn vulkan_waiver_does_not_cover_a_real_bug() {
    let r = Kernel::from_unary_f32(&def("ref_k", relu()));
    let c = Kernel::from_unary_f32_slang(&def("cand_k", input(0).unary(UnaryOp::Abs)));
    let rep = compare_unary_f32_vulkan(
        &work(),
        "vk_abs",
        0,
        &r,
        &c,
        Options {
            waive: negative_denormal,
            ..Options::exhaustive(Tier::Exact)
        },
    )
    .unwrap();
    assert!(!rep.accepted() && rep.mismatches > 0, "{rep:?}");
}

/// The flag takes effect: the SPIR-V declares `DenormPreserve` with it and not without. This is
/// what stops the next test passing (or failing) because `slangc` ignored its arguments.
#[test]
fn slangc_denorm_flag_reaches_the_spirv() {
    let c = Kernel::from_unary_f32_slang(&def("cand_k", relu()));
    let plain = slang_to_spirv(&work(), "vk_flag0", &c.source, &c.name, &[]).unwrap();
    let kept = slang_to_spirv(&work(), "vk_flag1", &c.source, &c.name, DENORM_PRESERVE).unwrap();
    assert!(!declares_denorm_preserve(&plain));
    assert!(declares_denorm_preserve(&kept));
}

/// With `DenormPreserve` the Slang relu is EXACT against cpu-c on every device: no waiver, and
/// `waived` stays 0. (NVIDIA reports `shaderDenormPreserveFloat32 = false` yet honours it; the
/// property is not a reliable gate there, so this measurement is the arbiter.)
#[test]
fn vulkan_relu_is_exact_with_denorm_preserve_on_every_device() {
    let r = Kernel::from_unary_f32(&def("ref_k", relu()));
    let c = Kernel::from_unary_f32_slang(&def("cand_k", relu()));
    let only: Option<usize> = std::env::var("UNPOPPED_VULKAN_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok());
    let mut failed = Vec::new();
    for (i, name) in vulkan_devices().unwrap().iter().enumerate() {
        if only.is_some_and(|o| o != i) {
            continue;
        }
        let opts = Options {
            slangc_args: DENORM_PRESERVE,
            ..Options::exhaustive(Tier::Exact)
        };
        let tag = format!("vk_relu_dp_{i}");
        let rep = compare_unary_f32_vulkan(&work(), &tag, i, &r, &c, opts).unwrap();
        println!("{name}: {rep:?}");
        if !rep.accepted() || rep.waived != 0 {
            failed.push(format!("{name}: {rep:?}"));
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}
