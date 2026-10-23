// SPDX-License-Identifier: MIT OR Apache-2.0
//! The harness's half of the ledger: the kernel hash, the emitter pins, coverage from a `Report`,
//! and the file round trip through Unpopped's schema. The schema itself is tested in Unpopped
//! (`unpopped::exactness`); this file does not restate it. No device needed; the Vulkan round trip
//! is in `vulkan_target.rs`.

use unpopped::exactness::{
    Coverage, CoverageKind, DeviceStamp, ExactnessLedger, ExactnessRecord, ExactnessVerdict,
    Finding, Withheld,
};
use unpopped::ir::{ScalarExpr, UnaryOp};
use unpopped_harness::ledger::{
    coverage_of, emitter_version, kernel_hash, read, write, EMITTER_VERSIONS,
};
use unpopped_harness::Report;
use unpopped_vocab::{ElementKind, HwStamp, TargetId};

fn relu() -> ScalarExpr {
    ScalarExpr::Unary(UnaryOp::Relu, Box::new(ScalarExpr::Input(0)))
}

fn stamp(compiler_version: &str) -> DeviceStamp {
    DeviceStamp::new(
        HwStamp {
            target: TargetId::parse("vulkan:sg64.ops-abr.arith-f16.cm-none").unwrap(),
            device_name: "AMD Radeon 610M".into(),
            runtime_version: "1.3.280".into(),
            captured_unix_s: 1_790_000_000,
        },
        "vendor=0x1002 driver=0x8000",
        compiler_version,
        vec!["-denorm-mode-fp32".into(), "preserve".into()],
    )
}

fn record(compiler_version: &str, hash: &str) -> ExactnessRecord {
    ExactnessRecord::new(
        stamp(compiler_version),
        &relu(),
        ElementKind::F32,
        ExactnessVerdict::Exact,
        hash,
        emitter_version("unpopped-slang"),
        Coverage::exhaustive(1 << 32),
    )
}

/// The file the harness writes is read back by Unpopped's own parser, record for record, and an
/// unreadable line is counted with its number. Then the staleness controls: an `Exact` measured
/// against another kernel or another compiler is not believed. Control: the matching lookup admits.
#[test]
fn a_ledger_round_trips_and_a_stale_exact_is_not_believed() {
    let mut ledger = ExactnessLedger::new();
    ledger.push(record("2026.8", "sha256:old"));
    let path = std::env::temp_dir().join(format!(
        "unpopped_harness_ledger_{}.jsonl",
        std::process::id()
    ));
    write(&ledger, &path).unwrap();
    let mut text = std::fs::read_to_string(&path).unwrap();
    text.push_str("not json at all\n");
    std::fs::write(&path, text).unwrap();
    let got = read(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(got.ledger, ledger, "the record survives the file");
    assert_eq!(
        got.rejected.len(),
        1,
        "the bad line is counted, not dropped"
    );
    assert_eq!(got.rejected[0].0, 2);

    let l = &got.ledger;
    let ask =
        |compiler: &str, hash: &str| l.find(&stamp(compiler), hash, &relu(), ElementKind::F32);
    assert_eq!(
        ask("2026.8", "sha256:old"),
        Finding::Exact,
        "the control admits"
    );
    assert_eq!(
        ask("2026.8", "sha256:new"),
        Finding::Withheld(Withheld::StaleKernel),
        "re-emitted kernel: not Exact"
    );
    assert_eq!(
        ask("2026.9", "sha256:old"),
        Finding::Withheld(Withheld::Unmeasured),
        "same kernel, upgraded slangc: another stamp, nothing measured"
    );
}

#[test]
fn coverage_is_what_the_report_says() {
    let exhaustive = Report {
        checked: 1 << 32,
        exhaustive: true,
        ..Report::default()
    };
    let c = coverage_of(&exhaustive, None);
    assert_eq!(
        (c.kind, c.tuples, c.seed),
        (CoverageKind::Exhaustive, 1 << 32, None)
    );
    let sampled = Report {
        checked: 4096,
        ..Report::default()
    };
    let c = coverage_of(&sampled, Some(7));
    assert_eq!(
        (c.kind, c.tuples, c.seed),
        (CoverageKind::Sampled, 4096, Some(7))
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
