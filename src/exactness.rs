// SPDX-License-Identifier: MIT OR Apache-2.0
//! Turn a [`Report`] into the record Unpopped reads: what this run, on this device, says about
//! this body (`unpopped::exactness`).
//!
//! The harness is the only producer of those records. Unpopped never probes a device, and the
//! Vulkan property bits are not evidence (NVIDIA reports `shaderDenormPreserveFloat32 = false`
//! yet honours the mode), so the verdict comes from the comparison itself.
//!
//! Coverage is exactly what was run. Exhaustive 2^32 exists here only for unary f32 bodies
//! ([`unary_f32_record`]); a binary or f64 body gets a record only from a measured run of its own,
//! and a sampled run can never produce [`ExactnessVerdict::Exact`].

use unpopped::exactness::{DeviceStamp, ExactnessRecord, ExactnessVerdict};
use unpopped::ir::OpDef;
use unpopped_vocab::ElementKind;

use crate::Report;

/// What a run of a unary f32 kernel concluded, or `None` if it concluded nothing.
///
/// - `Exact` only from an accepted exhaustive run with nothing waived: a waiver is a difference we
///   chose not to count, and a waived difference is still a difference on this device.
/// - `Flushes` from any observed difference, waived or not, and from a kernel that left outputs
///   unwritten (such a kernel is not exact; this is the conservative reading).
/// - `None` from a sampled run that saw nothing wrong: a sample can never accept.
#[must_use]
pub fn verdict_of(rep: &Report) -> Option<ExactnessVerdict> {
    let differed =
        rep.mismatches > 0 || rep.waived > 0 || rep.ref_never_wrote > 0 || rep.cand_never_wrote > 0;
    if differed {
        Some(ExactnessVerdict::Flushes)
    } else if rep.accepted() {
        Some(ExactnessVerdict::Exact)
    } else {
        None
    }
}

/// The record for a unary f32 `op` measured by `rep` on `stamp`, or `None` if `rep` concluded
/// nothing (see [`verdict_of`]).
#[must_use]
pub fn unary_f32_record(stamp: &DeviceStamp, op: &OpDef, rep: &Report) -> Option<ExactnessRecord> {
    let verdict = verdict_of(rep)?;
    Some(ExactnessRecord::new(
        stamp.clone(),
        &op.body,
        ElementKind::F32,
        verdict,
    ))
}

/// The record for an `op` of element kind `kind` measured by the sampled run `rep` on `stamp`
/// (the f64 screens in [`crate::double`], the binary and ternary f32 screens), or `None` if `rep`
/// concluded nothing. A sample never yields `Exact`.
#[must_use]
pub fn sampled_record(
    stamp: &DeviceStamp,
    op: &OpDef,
    kind: ElementKind,
    rep: &Report,
) -> Option<ExactnessRecord> {
    let verdict = verdict_of(rep)?;
    Some(ExactnessRecord::new(stamp.clone(), &op.body, kind, verdict))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean(exhaustive: bool) -> Report {
        Report {
            checked: 1 << 32,
            exhaustive,
            ..Report::default()
        }
    }

    #[test]
    fn an_accepted_exhaustive_run_is_exact() {
        assert_eq!(verdict_of(&clean(true)), Some(ExactnessVerdict::Exact));
    }

    /// The known-bad controls: each way a run can fall short of Exact.
    #[test]
    fn a_difference_of_any_kind_is_flushes() {
        for (name, rep) in [
            (
                "mismatch",
                Report {
                    mismatches: 1,
                    ..clean(true)
                },
            ),
            (
                "waived",
                Report {
                    waived: 8_388_607,
                    ..clean(true)
                },
            ),
            (
                "ref never wrote",
                Report {
                    ref_never_wrote: 1,
                    ..clean(true)
                },
            ),
            (
                "cand never wrote",
                Report {
                    cand_never_wrote: 1,
                    ..clean(true)
                },
            ),
            (
                "sampled mismatch",
                Report {
                    mismatches: 1,
                    ..clean(false)
                },
            ),
        ] {
            assert_eq!(verdict_of(&rep), Some(ExactnessVerdict::Flushes), "{name}");
        }
    }

    #[test]
    fn a_clean_sample_concludes_nothing() {
        assert_eq!(verdict_of(&clean(false)), None);
    }
}
