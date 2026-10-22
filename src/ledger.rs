// SPDX-License-Identifier: MIT OR Apache-2.0
//! The exactness ledger on disk: one JSON object per line, each a measurement and what it was
//! measured against (`docs/design-per-device-decline.md` in Unpopped, section 4).
//!
//! An `ExactnessRecord` is keyed by body and device stamp, but an `Exact` is a statement about a
//! kernel *as compiled*. So a line also names the compiler version, the hash of the kernel that ran
//! (SPIR-V for Vulkan, source for cpu-c), the emitter that produced it and how much was covered.
//! Unpopped 0.20.x has no fields for those, so until the 0.21 record carries them they live here,
//! and [`Ledger::current_for`] applies them: an entry whose compiler version or kernel hash differs
//! from what is about to run is dropped, and the drop is counted, never silent.
//!
//! # Format
//!
//! JSON Lines, `schema` first. An unknown `schema` or a malformed line is a typed [`LineError`]
//! that [`Ledger::parse`] collects with its line number; it is never a panic and never dropped
//! without being counted. Fields this version does not know are kept in [`LedgerEntry::extra`] and
//! written back, so a newer writer's additions survive a round trip through an older reader.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use unpopped::exactness::{DeviceStamp, ExactnessLedger, ExactnessRecord, ExactnessVerdict};
use unpopped_vocab::{HwStamp, TargetId};

use crate::Report;

/// The `schema` value this version writes and reads.
pub const SCHEMA: &str = "unpopped-exactness-ledger/1";

/// Versions of the emitters the harness lowers through, as pinned in `Cargo.lock`. A test reads the
/// lockfile and fails if one of these drifts, so the string cannot go stale unnoticed.
/// (Unpopped has no way to ask a backend its version; that is a 0.21 candidate.)
pub const EMITTER_VERSIONS: [(&str, &str); 3] = [
    ("unpopped-cpu-c", "0.20.19"),
    ("unpopped-slang", "0.20.19"),
    ("baracuda-cuda-emit", "0.20.1"),
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

/// How much of the input space the run covered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverageKind {
    Exhaustive,
    Sampled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coverage {
    pub kind: CoverageKind,
    /// Inputs (or tuples) compared.
    pub tuples: u64,
    /// The sampler's seed, for a sampled run that has one.
    pub seed: Option<u64>,
}

impl Coverage {
    /// What `rep` says it covered. `exhaustive` is the report's own word; a stride screen is
    /// `Sampled`.
    #[must_use]
    pub fn of_report(rep: &Report, seed: Option<u64>) -> Coverage {
        Coverage {
            kind: if rep.exhaustive {
                CoverageKind::Exhaustive
            } else {
                CoverageKind::Sampled
            },
            tuples: rep.checked,
            seed,
        }
    }
}

/// What a measurement was taken against, beyond the record's own stamp.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    /// The compiler's own version (`slangc -v`), which the stamp's arguments do not carry.
    pub compiler_version: String,
    /// [`kernel_hash`] of the kernel that ran.
    pub kernel_hash: String,
    /// [`emitter_version`] of the emitter that produced it.
    pub emitter_version: String,
    pub coverage: Coverage,
}

/// One line of the ledger.
#[derive(Clone, Debug, PartialEq)]
pub struct LedgerEntry {
    pub record: ExactnessRecord,
    pub provenance: Provenance,
    /// Fields this version does not know, kept so they are written back.
    pub extra: BTreeMap<String, Value>,
}

impl LedgerEntry {
    #[must_use]
    pub fn new(record: ExactnessRecord, provenance: Provenance) -> LedgerEntry {
        LedgerEntry {
            record,
            provenance,
            extra: BTreeMap::new(),
        }
    }

    /// The line, without its newline. `schema` is first.
    ///
    /// # Panics
    /// If the verdict is one this version does not know how to spell (`ExactnessVerdict` is
    /// `#[non_exhaustive]`): writing a guess would corrupt the ledger.
    #[must_use]
    pub fn to_line(&self) -> String {
        let st = &self.record.stamp;
        let verdict = match self.record.verdict {
            ExactnessVerdict::Exact => "exact",
            ExactnessVerdict::Flushes => "flushes",
            v => panic!("no spelling for verdict {v:?}; add one before writing it"),
        };
        let p = &self.provenance;
        // Fixed order: serde_json's map is sorted, so the line is assembled field by field.
        let mut fields: Vec<(&str, Value)> = vec![
            ("schema", json!(SCHEMA)),
            ("target", json!(st.hw.target.as_str())),
            ("device_name", json!(st.hw.device_name)),
            ("runtime_version", json!(st.hw.runtime_version)),
            ("driver_version", json!(st.driver_version)),
            ("compiler_args", json!(st.compiler_args)),
            ("compiler_version", json!(p.compiler_version)),
            ("key", json!(self.record.key)),
            ("verdict", json!(verdict)),
            ("kernel_hash", json!(p.kernel_hash)),
            ("emitter_version", json!(p.emitter_version)),
            (
                "coverage",
                json!({
                    "kind": match p.coverage.kind {
                        CoverageKind::Exhaustive => "exhaustive",
                        CoverageKind::Sampled => "sampled",
                    },
                    "tuples": p.coverage.tuples,
                    "seed": p.coverage.seed,
                }),
            ),
            ("captured_unix_s", json!(st.hw.captured_unix_s)),
        ];
        fields.extend(self.extra.iter().map(|(k, v)| (k.as_str(), v.clone())));
        let body: Vec<String> = fields
            .into_iter()
            .map(|(k, v)| format!("{}:{v}", Value::from(k)))
            .collect();
        format!("{{{}}}", body.join(","))
    }

    /// Parse one line.
    pub fn parse_line(line: &str) -> Result<LedgerEntry, LineError> {
        let v: Value = serde_json::from_str(line).map_err(|e| LineError::NotJson(e.to_string()))?;
        let Value::Object(mut m) = v else {
            return Err(LineError::NotJson("not an object".into()));
        };
        match m.remove("schema") {
            Some(Value::String(s)) if s == SCHEMA => {}
            Some(Value::String(s)) => return Err(LineError::UnknownSchema(s)),
            _ => return Err(LineError::Field("schema")),
        }
        let text = |m: &mut Map<String, Value>, f: &'static str| match m.remove(f) {
            Some(Value::String(s)) => Ok(s),
            _ => Err(LineError::Field(f)),
        };
        let target_text = text(&mut m, "target")?;
        let target = TargetId::parse(&target_text).map_err(|_| LineError::Field("target"))?;
        let device_name = text(&mut m, "device_name")?;
        let runtime_version = text(&mut m, "runtime_version")?;
        let driver_version = text(&mut m, "driver_version")?;
        let compiler_args = match m.remove("compiler_args") {
            Some(Value::Array(a)) => a
                .into_iter()
                .map(|x| match x {
                    Value::String(s) => Ok(s),
                    _ => Err(LineError::Field("compiler_args")),
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(LineError::Field("compiler_args")),
        };
        let compiler_version = text(&mut m, "compiler_version")?;
        let key = text(&mut m, "key")?;
        let verdict = match text(&mut m, "verdict")?.as_str() {
            "exact" => ExactnessVerdict::Exact,
            "flushes" => ExactnessVerdict::Flushes,
            _ => return Err(LineError::Field("verdict")),
        };
        let kernel_hash = text(&mut m, "kernel_hash")?;
        let emitter_version = text(&mut m, "emitter_version")?;
        let coverage = match m.remove("coverage") {
            Some(Value::Object(mut c)) => {
                let kind = match c.remove("kind") {
                    Some(Value::String(s)) if s == "exhaustive" => CoverageKind::Exhaustive,
                    Some(Value::String(s)) if s == "sampled" => CoverageKind::Sampled,
                    _ => return Err(LineError::Field("coverage.kind")),
                };
                let tuples = c
                    .remove("tuples")
                    .and_then(|t| t.as_u64())
                    .ok_or(LineError::Field("coverage.tuples"))?;
                let seed = match c.remove("seed") {
                    None | Some(Value::Null) => None,
                    Some(s) => Some(s.as_u64().ok_or(LineError::Field("coverage.seed"))?),
                };
                Coverage { kind, tuples, seed }
            }
            _ => return Err(LineError::Field("coverage")),
        };
        let captured_unix_s = m
            .remove("captured_unix_s")
            .and_then(|t| t.as_u64())
            .ok_or(LineError::Field("captured_unix_s"))?;
        Ok(LedgerEntry {
            record: ExactnessRecord {
                stamp: DeviceStamp::new(
                    HwStamp {
                        target,
                        device_name,
                        runtime_version,
                        captured_unix_s,
                    },
                    driver_version,
                    compiler_args,
                ),
                key,
                verdict,
            },
            provenance: Provenance {
                compiler_version,
                kernel_hash,
                emitter_version,
                coverage,
            },
            extra: m.into_iter().collect(),
        })
    }
}

/// Why a line was not read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LineError {
    NotJson(String),
    /// A well-formed line from a schema this version does not know.
    UnknownSchema(String),
    /// A required field is missing or has the wrong type.
    Field(&'static str),
}

/// Entries read, and the lines that were not.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ledger {
    pub entries: Vec<LedgerEntry>,
    /// `(1-based line number, why)` for every non-blank line that was not read.
    pub rejected: Vec<(usize, LineError)>,
}

/// What [`Ledger::current_for`] kept and what it set aside.
#[derive(Clone, Debug, PartialEq)]
pub struct Applied {
    pub ledger: ExactnessLedger,
    pub kept: usize,
    /// Entries for a different compiler version or kernel hash.
    pub stale: usize,
}

impl Ledger {
    /// Read JSON Lines. Blank lines are skipped; every other unreadable line is in `rejected`.
    #[must_use]
    pub fn parse(text: &str) -> Ledger {
        let mut l = Ledger::default();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match LedgerEntry::parse_line(line) {
                Ok(e) => l.entries.push(e),
                Err(e) => l.rejected.push((i + 1, e)),
            }
        }
        l
    }

    #[must_use]
    pub fn to_jsonl(&self) -> String {
        self.entries.iter().fold(String::new(), |mut s, e| {
            s.push_str(&e.to_line());
            s.push('\n');
            s
        })
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_jsonl()).map_err(|e| format!("write {}: {e}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Ledger, String> {
        std::fs::read_to_string(path)
            .map(|t| Ledger::parse(&t))
            .map_err(|e| format!("read {}: {e}", path.display()))
    }

    /// The Unpopped ledger for a kernel about to run with `compiler_version` and `kernel_hash`:
    /// only entries measured against exactly those. Until Unpopped's record carries the
    /// provenance, this is where a stale `Exact` is kept from being believed.
    #[must_use]
    pub fn current_for(&self, compiler_version: &str, kernel_hash: &str) -> Applied {
        let mut ledger = ExactnessLedger::new();
        let (mut kept, mut stale) = (0, 0);
        for e in &self.entries {
            if e.provenance.compiler_version == compiler_version
                && e.provenance.kernel_hash == kernel_hash
            {
                ledger.push(e.record.clone());
                kept += 1;
            } else {
                stale += 1;
            }
        }
        Applied {
            ledger,
            kept,
            stale,
        }
    }
}
