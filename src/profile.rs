// SPDX-License-Identifier: MIT OR Apache-2.0
//! Harness B and C: profile Unpopped's own stages across a corpus, then map the hotspots.
//!
//! B: for each unary op, fingerprint (SHA-256 of the IR text and of each emitter's source) and time
//! the stages that need no device: IR text round trip, then `generate` per emitter. Timings are the
//! minimum of `reps` runs, so a scheduler hiccup cannot inflate a row.
//!
//! C: raw nanoseconds are not comparable across emitters, so each emit time is divided by that
//! emitter's own median over the corpus. An op whose ratio exceeds [`HOTSPOT_RATIO`] on one
//! emitter while the others stay near 1.0 is where that emitter is the outlier.

use std::time::Instant;

use unpopped::ir::{input, OpDef};
use unpopped::{op_from_text, op_to_text, try_generate, Backend};
use unpopped_vocab::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

use crate::bench::UNARY_OPS;
use crate::ledger::kernel_hash;
use crate::CHUNK;

/// An op is a hotspot on an emitter when its emit time is at least this multiple of that
/// emitter's median emit time.
pub const HOTSPOT_RATIO: f64 = 3.0;

/// The emitters profiled, by name.
pub const EMITTERS: &[&str] = &["cpu-c", "cuda", "slang"];

#[derive(Clone, Debug, PartialEq)]
pub struct EmitterRow {
    pub emitter: &'static str,
    /// `sha256:` of the emitted source; `None` when the emitter declined.
    pub source_hash: Option<String>,
    pub source_bytes: usize,
    /// The emitter's decline text, when it declined. A decline is a finding, not a failure.
    pub declined: Option<String>,
    /// Time to emit, or to decline.
    pub emit_ns: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpProfile {
    pub op: String,
    /// `sha256:` of the IR text.
    pub ir_hash: String,
    pub ir_bytes: usize,
    /// IR text serialize + parse, nanoseconds.
    pub text_roundtrip_ns: u64,
    pub emitters: Vec<EmitterRow>,
}

fn backend(name: &str) -> Box<dyn Backend> {
    match name {
        "cpu-c" => Box::new(unpopped_cpu_c::CpuC),
        "cuda" => Box::new(baracuda_cuda_emit::Cuda),
        "slang" => Box::new(unpopped_slang::Slang),
        other => panic!("unknown emitter {other}"),
    }
}

fn min_ns(reps: u32, mut f: impl FnMut()) -> u64 {
    (0..reps.max(1))
        .map(|_| {
            let t = Instant::now();
            f();
            u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX)
        })
        .min()
        .expect("reps.max(1) >= 1")
}

/// Profile every unary op in [`UNARY_OPS`] on every emitter in [`EMITTERS`].
pub fn profile_unary(reps: u32) -> Vec<OpProfile> {
    let d = OperandDesc::new(1, &[CHUNK as i64 - 1], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[d, d], ArchSku::Sm89);
    UNARY_OPS
        .iter()
        .map(|(name, uop)| {
            let def = OpDef::elementwise(
                &format!("def_{name}"),
                1,
                &[ElementKind::F32],
                input(0).unary(*uop),
            );
            let text = op_to_text(&def);
            let round = min_ns(reps, || {
                let back = op_from_text(&op_to_text(&def)).expect("own text parses");
                assert_eq!(back, def, "text round trip must be lossless");
            });
            let emitters = EMITTERS
                .iter()
                .map(|e| {
                    let b = backend(e);
                    let first = try_generate(&def, &key, b.as_ref());
                    let emit_ns = min_ns(reps, || {
                        let _ = std::hint::black_box(try_generate(&def, &key, b.as_ref()));
                    });
                    let (source_hash, source_bytes, declined) = match first {
                        Ok(k) => (Some(kernel_hash(k.source.as_bytes())), k.source.len(), None),
                        Err(err) => (None, 0, Some(err.to_string())),
                    };
                    EmitterRow {
                        emitter: e,
                        source_hash,
                        source_bytes,
                        declined,
                        emit_ns,
                    }
                })
                .collect();
            OpProfile {
                op: (*name).to_string(),
                ir_hash: kernel_hash(text.as_bytes()),
                ir_bytes: text.len(),
                text_roundtrip_ns: round,
                emitters,
            }
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hotspot {
    pub op: String,
    pub emitter: &'static str,
    /// Emit time divided by this emitter's median emit time.
    pub ratio: f64,
}

fn median(mut v: Vec<u64>) -> f64 {
    v.sort_unstable();
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2] as f64
    } else {
        (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.0
    }
}

/// Harness C: every (op, emitter) whose emit time is at least `threshold` times that emitter's
/// median, worst first.
pub fn hotspots(profiles: &[OpProfile], threshold: f64) -> Vec<Hotspot> {
    let mut out = Vec::new();
    for e in EMITTERS {
        let times: Vec<u64> = profiles
            .iter()
            .filter_map(|p| p.emitters.iter().find(|r| r.emitter == *e))
            .filter(|r| r.declined.is_none())
            .map(|r| r.emit_ns)
            .collect();
        let med = median(times);
        if med <= 0.0 {
            continue;
        }
        for p in profiles {
            if let Some(r) = p
                .emitters
                .iter()
                .find(|r| r.emitter == *e && r.declined.is_none())
            {
                let ratio = r.emit_ns as f64 / med;
                if ratio >= threshold {
                    out.push(Hotspot {
                        op: p.op.clone(),
                        emitter: e,
                        ratio,
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| b.ratio.total_cmp(&a.ratio));
    out
}
