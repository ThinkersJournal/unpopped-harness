// SPDX-License-Identifier: MIT OR Apache-2.0
//! The harness's half of the exactness ledger: what only the producer can know.
//!
//! The record, its JSON Lines schema and the admission rules are Unpopped's
//! (`unpopped::exactness`, 0.21): the schema is owned there and nowhere else. This module supplies
//! the inputs a record needs that Unpopped cannot compute because it never runs a kernel: the hash
//! of what ran, the emitter's version, how much a [`Report`] covered, and reading and writing the
//! file.

use std::fmt::Write as _;
use std::path::Path;

use sha2::{Digest, Sha256};
use unpopped::exactness::{Coverage, ExactnessLedger, LedgerRead};

use crate::Report;

/// Versions of the emitters the harness lowers through, as pinned in `Cargo.lock`. A test reads the
/// lockfile and fails if one of these drifts, so the string cannot go stale unnoticed.
/// (`emitter_version` stays a string the producer supplies: Unpopped cannot ask a `Backend`.)
pub const EMITTER_VERSIONS: [(&str, &str); 3] = [
    ("unpopped-cpu-c", "0.22.0"),
    ("unpopped-slang", "0.22.0"),
    ("baracuda-cuda-emit", "0.22.0"),
];

/// `"{crate} {version}"` for one of [`EMITTER_VERSIONS`].
///
/// # Panics
/// If `crate_name` is not one of the emitters the harness uses.
#[must_use]
pub fn emitter_version(crate_name: &str) -> String {
    let (n, v) = EMITTER_VERSIONS
        .iter()
        .find(|(n, _)| *n == crate_name)
        .unwrap_or_else(|| panic!("{crate_name} is not an emitter this harness uses"));
    format!("{n} {v}")
}

/// `sha256:<hex>` of what actually ran: the SPIR-V bytes for Vulkan, the source text for cpu-c.
#[must_use]
pub fn kernel_hash(bytes: &[u8]) -> String {
    let mut s = String::from("sha256:");
    for b in Sha256::digest(bytes) {
        write!(s, "{b:02x}").expect("writing to a String");
    }
    s
}

/// What `rep` says it covered. `exhaustive` is the report's own word; a stride screen is sampled.
#[must_use]
pub fn coverage_of(rep: &Report, seed: Option<u64>) -> Coverage {
    if rep.exhaustive {
        Coverage::exhaustive(rep.checked)
    } else {
        Coverage::sampled(rep.checked, seed)
    }
}

/// Write `ledger` as JSON Lines.
pub fn write(ledger: &ExactnessLedger, path: &Path) -> Result<(), String> {
    std::fs::write(path, ledger.to_jsonl()).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Read a JSON Lines ledger. Lines that were not read are in [`LedgerRead::rejected`].
pub fn read(path: &Path) -> Result<LedgerRead, String> {
    std::fs::read_to_string(path)
        .map(|t| ExactnessLedger::parse_jsonl(&t))
        .map_err(|e| format!("read {}: {e}", path.display()))
}
