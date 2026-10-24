// SPDX-License-Identifier: MIT OR Apache-2.0
//! C14 (Unpopped ROADMAP section 6): does a Vulkan device preserve NaN, infinities and signed zero
//! through a Slang kernel? The N2 ruling waits on this: KISS recommends gate-decline later, with no
//! spec text until some device is measured not preserving NaN.
//!
//! This is the behavioural measurement. `vulkane` 0.16 does not expose
//! `shaderSignedZeroInfNanPreserveFloat32`, so the device's *claim* is not read here; what a kernel
//! does is. Each probe is run exhaustively over all 2^32 f32 inputs against cpu-c and printed per
//! device. A device that fails a probe is the evidence the ruling needs; the asserts here only keep
//! the instrument honest (the planted control must be caught).
//!
//! Needs `slangc` and a Vulkan driver.

use std::path::PathBuf;
use unpopped::ir::{input, konst, BinaryOp, Expr, OpDef, UnaryOp};
use unpopped_harness::vulkan::{compare_unary_f32_vulkan, vulkan_devices, Options};
use unpopped_harness::{Kernel, Report, Tier};
use unpopped_vocab::ElementKind;

fn work() -> PathBuf {
    std::env::temp_dir().join("unpopped-harness-vulkan-nan")
}

fn def(name: &str, body: Expr) -> OpDef {
    OpDef::elementwise(name, 1, &[ElementKind::F32], body)
}

/// Denormal inputs (either sign, not zero). Devices flush these (seen on AMD 610M and RTX 4070
/// Laptop); that is a separate, already-recorded property, so it is waived and counted here to keep
/// the NaN/Inf/zero reading clean.
fn denormal(bits: u32) -> bool {
    bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0
}

struct Probe {
    name: &'static str,
    what: &'static str,
    body: fn() -> Expr,
}

fn probes() -> Vec<Probe> {
    vec![
        Probe {
            name: "isnan",
            what: "x != x: does a NaN operand reach the comparison",
            body: || input(0).binary(BinaryOp::CmpNe, input(0)),
        },
        Probe {
            name: "isinf",
            // `|x| > f32::MAX` is true only for an infinity. An infinity literal would be simpler but
            // the published `unpopped-slang` 0.22.0 spells it as a C macro `slangc` rejects (fixed in
            // Unpopped 0.22.14), so the probe avoids the literal.
            what: "|x| > f32::MAX: are infinities kept",
            body: || {
                input(0)
                    .unary(UnaryOp::Abs)
                    .binary(BinaryOp::CmpGt, konst(f64::from(f32::MAX)))
            },
        },
        Probe {
            // A plain `1/x` compare measured reciprocal precision instead (AMD 610M: 1-ULP
            // differences on 486M inputs), so the reciprocal is classified: +1 for +inf, -1 for
            // -inf, 0 otherwise. That differs only where the sign of zero or an infinity is lost.
            name: "zero_sign",
            what: "class(1/x) in {+inf,-inf,finite}: is the sign of zero kept",
            body: || {
                let r = || input(0).unary(UnaryOp::Recip);
                r().binary(BinaryOp::CmpGt, konst(f64::from(f32::MAX)))
                    - r().binary(BinaryOp::CmpLt, konst(-f64::from(f32::MAX)))
            },
        },
    ]
}

fn run(
    device: usize,
    tag: &str,
    reference: &Expr,
    candidate: &Expr,
    name: &str,
) -> Result<Report, String> {
    let r = Kernel::from_unary_f32(&def(&format!("ref_{name}"), reference.clone()));
    let c = Kernel::from_unary_f32_slang(&def(&format!("cand_{name}"), candidate.clone()));
    compare_unary_f32_vulkan(
        &work(),
        tag,
        device,
        &r,
        &c,
        Options {
            waive: denormal,
            ..Options::exhaustive(Tier::Exact)
        },
    )
}

/// The measurement. Prints one line per (device, probe); it does not assert a verdict, because the
/// verdict is the finding. It does fail if the instrument cannot run at all.
#[test]
fn nan_inf_and_signed_zero_survival_on_every_vulkan_device() {
    let only: Option<usize> = std::env::var("UNPOPPED_VULKAN_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok());
    let want = std::env::var("UNPOPPED_PROBE").ok();
    let mut ran = 0;
    for (i, dev) in vulkan_devices().unwrap().iter().enumerate() {
        if only.is_some_and(|o| o != i) {
            continue;
        }
        for p in probes() {
            if want.as_deref().is_some_and(|w| w != p.name) {
                continue;
            }
            let body = (p.body)();
            let rep = run(i, &format!("nan_{i}_{}", p.name), &body, &body, p.name)
                .unwrap_or_else(|e| panic!("{dev} {}: {e}", p.name));
            println!(
                "C14 {dev} | {:<5} | {} | checked {} mismatches {} waived(denormal) {} first {:?}",
                p.name, p.what, rep.checked, rep.mismatches, rep.waived, rep.first
            );
            ran += 1;
        }
    }
    assert!(ran > 0, "no Vulkan device ran a probe");
}

/// Control: the probe must be able to fail. A candidate that is NaN-blind (isnan always 0) differs
/// from the reference on every NaN input (2^24 - 2 patterns), so the compare must reject it.
#[test]
fn a_nan_blind_candidate_is_rejected_by_the_isnan_probe() {
    let reference = input(0).binary(BinaryOp::CmpNe, input(0));
    let blind = konst(0.0) * input(0) + konst(0.0); // not a NaN test: NaN in, NaN out
    let blind = blind.binary(BinaryOp::CmpEq, konst(1.0)); // never true
    let rep = run(0, "nan_ctl", &reference, &blind, "ctl").unwrap();
    assert!(
        !rep.accepted() && rep.mismatches >= (1 << 23) - 1,
        "the isnan probe did not see a NaN-blind kernel: {rep:?}"
    );
}
