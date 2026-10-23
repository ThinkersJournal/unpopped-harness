// SPDX-License-Identifier: MIT OR Apache-2.0
//! The ledger file: what it writes, what it refuses, and that a stale `Exact` is not believed.
//! No device needed; the Vulkan round trip is in `vulkan_target.rs`.

use serde_json::json;
use unpopped::exactness::{DeviceStamp, ExactnessRecord, ExactnessVerdict};
use unpopped::ir::{ScalarExpr, UnaryOp};
use unpopped_harness::ledger::{
    emitter_version, kernel_hash, Coverage, CoverageKind, Ledger, LedgerEntry, LineError,
    Provenance, EMITTER_VERSIONS, SCHEMA,
};
use unpopped_harness::Report;
use unpopped_vocab::{ElementKind, HwStamp, TargetId};

fn relu() -> ScalarExpr {
    ScalarExpr::Unary(UnaryOp::Relu, Box::new(ScalarExpr::Input(0)))
}

fn stamp() -> DeviceStamp {
    DeviceStamp::new(
        HwStamp {
            target: TargetId::parse("vulkan:sg64.ops-abr.arith-f16.cm-none").unwrap(),
            device_name: "AMD Radeon 610M".into(),
            runtime_version: "1.3.280".into(),
            captured_unix_s: 1_790_000_000,
        },
        "vendor=0x1002 driver=0x8000",
        vec!["-denorm-mode-fp32".into(), "preserve".into()],
    )
}

fn prov(hash: &str) -> Provenance {
    Provenance {
        compiler_version: "2026.8".into(),
        kernel_hash: hash.into(),
        emitter_version: emitter_version("unpopped-slang"),
        coverage: Coverage {
            kind: CoverageKind::Exhaustive,
            tuples: 1 << 32,
            seed: None,
        },
    }
}

fn entry(verdict: ExactnessVerdict, hash: &str) -> LedgerEntry {
    LedgerEntry::new(
        ExactnessRecord::new(stamp(), &relu(), ElementKind::F32, verdict),
        prov(hash),
    )
}

/// The line is a format other tools read: pin it, `schema` first.
#[test]
fn the_line_text_is_pinned() {
    let line = entry(ExactnessVerdict::Exact, "sha256:aa").to_line();
    assert_eq!(
        line,
        concat!(
            r#"{"schema":"unpopped-exactness-ledger/1","#,
            r#""target":"vulkan:sg64.ops-abr.arith-f16.cm-none","#,
            r#""device_name":"AMD Radeon 610M","runtime_version":"1.3.280","#,
            r#""driver_version":"vendor=0x1002 driver=0x8000","#,
            r#""compiler_args":["-denorm-mode-fp32","preserve"],"compiler_version":"2026.8","#,
            r#""key":"f32:{\"Unary\":[\"Relu\",{\"Input\":0}]}","verdict":"exact","#,
            r#""kernel_hash":"sha256:aa","emitter_version":"unpopped-slang 0.20.19","#,
            r#""coverage":{"kind":"exhaustive","seed":null,"tuples":4294967296},"#,
            r#""captured_unix_s":1790000000}"#
        )
    );
    assert!(line.starts_with(&format!(r#"{{"schema":"{SCHEMA}""#)));
}

#[test]
fn a_ledger_round_trips_and_keeps_fields_it_does_not_know() {
    let mut e = entry(ExactnessVerdict::Flushes, "sha256:bb");
    e.extra
        .insert("note".into(), json!({"from": "a newer writer"}));
    let l = Ledger {
        entries: vec![e, entry(ExactnessVerdict::Exact, "sha256:cc")],
        rejected: vec![],
    };
    let back = Ledger::parse(&l.to_jsonl());
    assert_eq!(back, l);
    assert_eq!(
        back.to_jsonl(),
        l.to_jsonl(),
        "a second write is the same text"
    );
}

/// Known-bad lines: each is rejected with its line number, the good lines around them are still
/// read, and nothing panics or vanishes.
#[test]
fn unreadable_lines_are_counted_not_dropped() {
    let good = entry(ExactnessVerdict::Exact, "sha256:aa").to_line();
    let unknown = good.replace(SCHEMA, "unpopped-exactness-ledger/2");
    let no_hash = good.replace(r#""kernel_hash":"sha256:aa","#, "");
    let bad_verdict = good.replace(r#""verdict":"exact""#, r#""verdict":"maybe""#);
    let bad_target = good.replace("vulkan:sg64.ops-abr.arith-f16.cm-none", "Not A Target");
    let text = format!(
        "{good}\n\nnot json at all\n{unknown}\n{no_hash}\n{bad_verdict}\n{bad_target}\n{good}\n"
    );
    let l = Ledger::parse(&text);
    assert_eq!(
        l.entries.len(),
        2,
        "the good lines around the bad ones are read"
    );
    let lines: Vec<usize> = l.rejected.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        lines,
        [3, 4, 5, 6, 7],
        "blank line 2 is skipped, the rest are counted"
    );
    assert!(matches!(l.rejected[0].1, LineError::NotJson(_)));
    assert_eq!(
        l.rejected[1].1,
        LineError::UnknownSchema("unpopped-exactness-ledger/2".into())
    );
    assert_eq!(l.rejected[2].1, LineError::Field("kernel_hash"));
    assert_eq!(l.rejected[3].1, LineError::Field("verdict"));
    assert_eq!(l.rejected[4].1, LineError::Field("target"));
}

/// An `Exact` measured against another kernel (or another compiler) is not believed for this one,
/// and the drop is counted. Control: the matching entry is admitted and Unpopped answers `Exact`.
#[test]
fn a_stale_exact_is_not_believed() {
    let l = Ledger {
        entries: vec![entry(ExactnessVerdict::Exact, "sha256:old")],
        rejected: vec![],
    };
    let ask = |a: &unpopped_harness::ledger::Applied| {
        a.ledger.lookup(&stamp(), &relu(), ElementKind::F32)
    };

    let same = l.current_for("2026.8", "sha256:old");
    assert_eq!((same.kept, same.stale), (1, 0));
    assert_eq!(
        ask(&same),
        Some(ExactnessVerdict::Exact),
        "the control admits"
    );

    let new_kernel = l.current_for("2026.8", "sha256:new");
    assert_eq!((new_kernel.kept, new_kernel.stale), (0, 1));
    assert_eq!(
        ask(&new_kernel),
        None,
        "re-emitted kernel: unmeasured, not Exact"
    );

    let new_compiler = l.current_for("2026.9", "sha256:old");
    assert_eq!((new_compiler.kept, new_compiler.stale), (0, 1));
    assert_eq!(
        ask(&new_compiler),
        None,
        "same source, upgraded slangc: unmeasured"
    );
}

#[test]
fn coverage_is_what_the_report_says() {
    let exhaustive = Report {
        checked: 1 << 32,
        exhaustive: true,
        ..Report::default()
    };
    assert_eq!(
        Coverage::of_report(&exhaustive, None),
        Coverage {
            kind: CoverageKind::Exhaustive,
            tuples: 1 << 32,
            seed: None
        }
    );
    let sampled = Report {
        checked: 4096,
        ..Report::default()
    };
    assert_eq!(
        Coverage::of_report(&sampled, Some(7)),
        Coverage {
            kind: CoverageKind::Sampled,
            tuples: 4096,
            seed: Some(7)
        }
    );
}

#[test]
fn kernel_hash_is_sha256() {
    assert_eq!(
        kernel_hash(b""),
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        kernel_hash(b"abc"),
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_ne!(kernel_hash(b"a"), kernel_hash(b"b"));
}

/// The emitter strings are written by hand, so the lockfile is their checker.
#[test]
fn emitter_versions_match_the_lockfile() {
    let lock = include_str!("../Cargo.lock");
    for (name, version) in EMITTER_VERSIONS {
        let want = format!("name = \"{name}\"\nversion = \"{version}\"");
        let lock = lock.replace("\r\n", "\n");
        assert!(
            lock.contains(&want),
            "{name} {version} is not what Cargo.lock pins: update EMITTER_VERSIONS"
        );
    }
}
