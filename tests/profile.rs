// SPDX-License-Identifier: MIT OR Apache-2.0
//! Harness B and C: device-free, so they run on any runner.

use unpopped_harness::bench::UNARY_OPS;
use unpopped_harness::profile::{hotspots, profile_unary, EmitterRow, OpProfile, EMITTERS};

#[test]
fn every_op_has_a_row_per_emitter_with_distinct_fingerprints() {
    let p = profile_unary(2);
    assert_eq!(p.len(), UNARY_OPS.len());
    for row in &p {
        assert_eq!(row.emitters.len(), EMITTERS.len(), "{}", row.op);
        assert!(row.ir_hash.starts_with("sha256:") && row.ir_bytes > 0);
        for e in &row.emitters {
            assert!(e.emit_ns > 0, "{} {}", row.op, e.emitter);
            // Exactly one of: emitted (hash, bytes) or declined (text).
            assert_eq!(e.source_hash.is_some(), e.declined.is_none());
            assert_eq!(
                e.source_bytes > 0,
                e.declined.is_none(),
                "{} {}",
                row.op,
                e.emitter
            );
        }
        // Emitters that emitted spell different languages: equal hashes would mean one was not run.
        let mut h: Vec<_> = row
            .emitters
            .iter()
            .filter_map(|e| e.source_hash.as_ref())
            .collect();
        let n = h.len();
        h.sort();
        h.dedup();
        assert_eq!(h.len(), n, "{}", row.op);
    }
    // Distinct ops must fingerprint differently.
    let mut ir: Vec<_> = p.iter().map(|r| &r.ir_hash).collect();
    ir.sort();
    ir.dedup();
    assert_eq!(ir.len(), p.len());
}

#[test]
fn fingerprints_are_deterministic() {
    let a = profile_unary(1);
    let b = profile_unary(1);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.ir_hash, y.ir_hash);
        for (ex, ey) in x.emitters.iter().zip(&y.emitters) {
            assert_eq!(ex.source_hash, ey.source_hash, "{} {}", x.op, ex.emitter);
            assert_eq!(ex.declined, ey.declined, "{} {}", x.op, ex.emitter);
        }
    }
}

fn synthetic(op: &str, ns: [u64; 3]) -> OpProfile {
    OpProfile {
        op: op.into(),
        ir_hash: String::new(),
        ir_bytes: 0,
        text_roundtrip_ns: 0,
        emitters: EMITTERS
            .iter()
            .zip(ns)
            .map(|(e, emit_ns)| EmitterRow {
                emitter: e,
                source_hash: Some(String::new()),
                source_bytes: 1,
                declined: None,
                emit_ns,
            })
            .collect(),
    }
}

/// Planted outlier: one op 10x slower on one emitter only must be found, and only there.
#[test]
fn hotspot_finds_the_planted_outlier_and_only_it() {
    let mut ps: Vec<_> = (0..9)
        .map(|i| synthetic(&format!("op{i}"), [100, 200, 300]))
        .collect();
    ps.push(synthetic("slow", [100, 2000, 300]));
    let h = hotspots(&ps, 3.0);
    assert_eq!(h.len(), 1, "{h:?}");
    assert_eq!((h[0].op.as_str(), h[0].emitter), ("slow", "cuda"));
    assert!((h[0].ratio - 10.0).abs() < 1e-9);
    // Control: a uniform corpus has none.
    let flat: Vec<_> = (0..5)
        .map(|i| synthetic(&format!("f{i}"), [5, 5, 5]))
        .collect();
    assert!(hotspots(&flat, 3.0).is_empty());
}

/// The corpus must contain a decline (Slang has no erf), and it must be reported, not hidden or
/// counted into the hotspot median.
#[test]
fn declines_are_recorded_and_excluded_from_hotspots() {
    let p = profile_unary(1);
    let erf = p
        .iter()
        .find(|r| r.op == "erf")
        .expect("erf is in the corpus");
    let slang = erf.emitters.iter().find(|e| e.emitter == "slang").unwrap();
    assert!(
        slang.declined.as_deref().is_some_and(|d| d.contains("erf")),
        "{slang:?}"
    );
    let relu = p.iter().find(|r| r.op == "relu").unwrap();
    assert!(
        relu.emitters.iter().all(|e| e.declined.is_none()),
        "control: relu emits everywhere"
    );
    assert!(hotspots(&p, 3.0)
        .iter()
        .all(|h| !(h.op == "erf" && h.emitter == "slang")));
}
