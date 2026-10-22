// SPDX-License-Identifier: MIT OR Apache-2.0
//! f64 comparison by stratified sampling, for unary, binary and ternary bodies. A sample can never
//! accept.
//!
//! 2^64 inputs cannot be walked, so every f64 run is a screen, not a proof: its verdict is
//! [`ExactnessVerdict::Flushes`](unpopped::exactness::ExactnessVerdict::Flushes) (a difference was
//! seen) or nothing at all (see [`crate::exactness::verdict_of`]). `Report::exhaustive` is always
//! `false` here, so `Report::accepted` is always `false` too.
//!
//! An operand's class is its sign and biased exponent field: 2 x 2048 = 4096 classes, which include
//! the zeros, every subnormal, every normal binade, the infinities and the NaNs. What a run
//! covers, by arity:
//!
//! - unary: every class receives [`SampleSpec::per_cell`] values;
//! - binary: every ordered pair of classes (4096 x 4096 cells) receives `per_cell` pairs;
//! - ternary: for each operand pair (0,1), (0,2), (1,2), every ordered class pair receives
//!   `per_cell` triples, the remaining operand's class drawn at random.
//!
//! Mantissas come from a seeded generator and from the corner mantissas (all-zero, all-one, the
//! quiet bit). A fixed grid of 16 edge values, crossed with itself `arity` times, is added on top.
//! The cell counts are read back from the generated operand bits, and the driver refuses a run that
//! left a cell to chance. A defect needing a particular mantissa inside one cell, or (ternary) all
//! three classes at once, is reached only by chance; that is why a clean sample concludes nothing.

use std::path::Path;
use std::process::Command;

use crate::{compile, Kernel, Report, Tier, CHUNK};

/// Operand classes: sign (2) x biased exponent field (2048).
pub const OPERAND_CLASSES: u64 = 4096;
/// Ordered pairs of operand classes.
pub const CLASS_PAIRS: u64 = OPERAND_CLASSES * OPERAND_CLASSES;
/// The fixed edge values, as bits; the grid crosses this list with itself.
pub const EDGE_VALUES: [u64; 16] = [
    0x0000_0000_0000_0000, // +0
    0x8000_0000_0000_0000, // -0
    0x0000_0000_0000_0001, // smallest subnormal
    0x8000_0000_0000_0001,
    0x000f_ffff_ffff_ffff, // largest subnormal
    0x800f_ffff_ffff_ffff,
    0x0010_0000_0000_0000, // smallest normal
    0x3ff0_0000_0000_0000, // 1
    0xbff0_0000_0000_0000, // -1
    0x7fef_ffff_ffff_ffff, // largest finite
    0xffef_ffff_ffff_ffff,
    0x7ff0_0000_0000_0000, // +inf
    0xfff0_0000_0000_0000, // -inf
    0x7ff8_0000_0000_0000, // quiet NaN
    0x7ff4_0000_0000_0001, // signalling NaN
    0xfff8_0000_0000_0000, // negative quiet NaN
];

/// Two distinct NaN patterns, as in the f32 driver: a cell counts as never-written only if it still
/// holds the sentinel after BOTH passes.
const SENTINEL_A: u64 = 0x7ff8_0000_dead_beef;
const SENTINEL_B: u64 = 0xfff8_0000_beef_dead;

/// How big the sample is and where its randomness comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleSpec {
    /// Tuples drawn for each class cell (see the module docs for what a cell is per arity).
    pub per_cell: u32,
    pub seed: u64,
}

impl SampleSpec {
    /// A default sized to the arity: 4096 cells cannot be thin, 16.7 million can.
    pub fn for_arity(arity: u32) -> SampleSpec {
        SampleSpec {
            per_cell: if arity == 1 { 64 } else { 1 },
            seed: 0x005e_ed0f,
        }
    }

    /// Class cells a run of this arity visits, summed over operand pairs.
    pub fn cells(arity: u32) -> u64 {
        match arity {
            1 => OPERAND_CLASSES,
            2 => CLASS_PAIRS,
            3 => 3 * CLASS_PAIRS,
            _ => panic!("arity {arity}: the f64 screens cover 1 to 3 operands"),
        }
    }

    /// Edge-grid tuples for this arity: 16^arity.
    pub fn edge_tuples(arity: u32) -> u64 {
        (EDGE_VALUES.len() as u64).pow(arity)
    }

    /// Tuples a run of this arity with this spec compares.
    pub fn tuples(&self, arity: u32) -> u64 {
        Self::cells(arity) * self.per_cell as u64 + Self::edge_tuples(arity)
    }
}

/// A sampled run: the comparison plus exactly what it covered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampledReport {
    /// `exhaustive` is `false`, so this never accepts. `first` is left empty (it holds f32 bits);
    /// the failing tuples are in [`SampledReport::first_failures`].
    pub report: Report,
    pub arity: u32,
    pub spec: SampleSpec,
    /// Class cells visited (always [`SampleSpec::cells`] on success).
    pub cells: u64,
    /// Edge-grid tuples compared in addition to the cells.
    pub edge_tuples: u64,
    /// Distinct classes (unary) or class pairs (binary; ternary: operand pairs (0,1), (0,2),
    /// (1,2)) present in the generated operand bits. Each is full by construction.
    pub covered: Vec<u64>,
    /// The first failing tuples.
    pub first_failures: Vec<Failure>,
}

/// One failing tuple, as bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// Operand bits, `arity` of them.
    pub operands: Vec<u64>,
    pub reference: u64,
    pub candidate: u64,
}

impl Kernel {
    fn from_f64(
        op: &unpopped::ir::OpDef,
        cat: unpopped_vocab::OpCategory,
        operands: usize,
    ) -> Kernel {
        use unpopped_vocab::{structure_key, ArchSku, ElementKind, OperandDesc};
        // Alignment 8 defeats vectorization, giving the scalar schedule.
        let d = OperandDesc::new(1, &[CHUNK as i64 - 1], &[1], ElementKind::F64, 8);
        let key = structure_key(cat, &vec![d; operands + 1], ArchSku::Sm89);
        let k = unpopped::generate(op, &key, &unpopped_cpu_c::CpuC);
        Kernel {
            name: k.name,
            source: k.source,
        }
    }

    /// Lower a unary f64 elementwise `OpDef` through the cpu-c emitter. The exported symbol is
    /// `name(const double* in0, double* out, long long n)`.
    pub fn from_unary_f64(op: &unpopped::ir::OpDef) -> Kernel {
        Self::from_f64(op, unpopped_vocab::OpCategory::UnaryElementwise, 1)
    }

    /// As [`Kernel::from_unary_f64`], binary: `name(const double* in0, const double* in1, double* out, long long n)`.
    pub fn from_binary_f64(op: &unpopped::ir::OpDef) -> Kernel {
        Self::from_f64(op, unpopped_vocab::OpCategory::BinaryElementwise, 2)
    }

    /// As [`Kernel::from_unary_f64`], ternary: three inputs, then the output.
    pub fn from_ternary_f64(op: &unpopped::ir::OpDef) -> Kernel {
        Self::from_f64(op, unpopped_vocab::OpCategory::TernaryElementwise, 3)
    }
}

fn edge_list() -> String {
    EDGE_VALUES
        .iter()
        .map(|b| format!("0x{b:016x}ull"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The C that fills `in0..in{arity-1}` for the tuples of one run, calling `emit()` after each.
fn generation(arity: u32, per_cell: u32) -> String {
    match arity {
        1 => format!(
            r#"    for (uint32_t c = 0; c < 4096u; ++c) {{
        for (uint32_t k = 0; k < {per_cell}u; ++k) {{
            uint64_t x0 = operand(c);
            in0[n] = f_of(x0);
            seen[0][cls(x0)] = 1;
            if (++n == CHUNK) {{ flush(n); n = 0; }}
        }}
        ++cells;
    }}
    for (int a = 0; a < 16; ++a) {{
        in0[n] = f_of(EDGE[a]); ++edge_tuples;
        if (++n == CHUNK) {{ flush(n); n = 0; }}
    }}
"#
        ),
        2 => format!(
            r#"    for (uint32_t ca = 0; ca < 4096u; ++ca) {{
        for (uint32_t cb = 0; cb < 4096u; ++cb) {{
            for (uint32_t k = 0; k < {per_cell}u; ++k) {{
                uint64_t x0 = operand(ca), x1 = operand(cb);
                in0[n] = f_of(x0); in1[n] = f_of(x1);
                seen[0][(uint64_t)cls(x0) * 4096u + cls(x1)] = 1;
                if (++n == CHUNK) {{ flush(n); n = 0; }}
            }}
            ++cells;
        }}
    }}
    for (int a = 0; a < 16; ++a) for (int b = 0; b < 16; ++b) {{
        in0[n] = f_of(EDGE[a]); in1[n] = f_of(EDGE[b]); ++edge_tuples;
        if (++n == CHUNK) {{ flush(n); n = 0; }}
    }}
"#
        ),
        3 => format!(
            r#"    /* Operand pair p: 0 = (0,1), 1 = (0,2), 2 = (1,2); the remaining operand gets a random class. */
    for (int p = 0; p < 3; ++p) {{
        for (uint32_t ca = 0; ca < 4096u; ++ca) {{
            for (uint32_t cb = 0; cb < 4096u; ++cb) {{
                for (uint32_t k = 0; k < {per_cell}u; ++k) {{
                    uint32_t cx = (uint32_t)(rng() & 4095u);
                    uint32_t c0 = (p == 2) ? cx : ca;
                    uint32_t c1 = (p == 0) ? cb : (p == 2 ? ca : cx);
                    uint32_t c2 = (p == 0) ? cx : cb;
                    uint64_t x0 = operand(c0), x1 = operand(c1), x2 = operand(c2);
                    in0[n] = f_of(x0); in1[n] = f_of(x1); in2[n] = f_of(x2);
                    {{
                        uint32_t u = (p == 2) ? cls(x1) : cls(x0);
                        uint32_t v = (p == 0) ? cls(x1) : cls(x2);
                        seen[p][(uint64_t)u * 4096u + v] = 1;
                    }}
                    if (++n == CHUNK) {{ flush(n); n = 0; }}
                }}
                ++cells;
            }}
        }}
    }}
    for (int a = 0; a < 16; ++a) for (int b = 0; b < 16; ++b) for (int c = 0; c < 16; ++c) {{
        in0[n] = f_of(EDGE[a]); in1[n] = f_of(EDGE[b]); in2[n] = f_of(EDGE[c]); ++edge_tuples;
        if (++n == CHUNK) {{ flush(n); n = 0; }}
    }}
"#
        ),
        _ => unreachable!(),
    }
}

fn driver(reference: &Kernel, cand: &Kernel, arity: u32, tier: Tier, spec: SampleSpec) -> String {
    let (tier_kind, tier_n) = match tier {
        Tier::Exact => (0, 0),
        Tier::Ulp(n) => (1, n),
    };
    let ins: String = (0..arity).map(|i| format!("in{i}, ")).collect();
    let decl: String = (0..arity).map(|i| format!("in{i}[CHUNK], ")).collect();
    let first_fmt = format!("first{}", " %llu".repeat(arity as usize + 2));
    let first_args: String = (0..arity)
        .map(|i| format!("(unsigned long long)b_of(in{i}[i]), "))
        .collect();
    // Seen-bitmaps: one of 4096 bytes (unary), one of 4096^2 (binary), three of 4096^2 (ternary).
    let (np, seen_len) = match arity {
        1 => (1, 4096u64),
        2 => (1, CLASS_PAIRS),
        _ => (3, CLASS_PAIRS),
    };
    let body = generation(arity, spec.per_cell);
    format!(
        r#"{ref_src}
{cand_src}
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#define CHUNK {chunk}
static double f_of(uint64_t b) {{ double f; memcpy(&f, &b, 8); return f; }}
static uint64_t b_of(double f) {{ uint64_t b; memcpy(&b, &f, 8); return b; }}
/* The f64 total order as a signed key: -0 and +0 are the same point, as in Tier::Ulp. */
static int64_t key_of(uint64_t b) {{
    return (b >> 63) ? -(int64_t)(b & 0x7fffffffffffffffull) : (int64_t)b;
}}
static uint64_t rng_state = {seed}ull;
static uint64_t rng(void) {{
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}}
static const uint64_t EDGE[16] = {{ {edges} }};
static const uint64_t CORNER[3] = {{ 0ull, 0xfffffffffffffull, 0x8000000000000ull }};
/* One operand of class c (sign<<11 | exponent field): a corner mantissa 3 times in 8, else random. */
static uint64_t operand(uint32_t c) {{
    uint64_t r = rng();
    uint64_t m = ((r & 7u) < 3u) ? CORNER[r & 7u] : ((r >> 8) & 0xfffffffffffffull);
    return ((uint64_t)(c >> 11) << 63) | ((uint64_t)(c & 2047u) << 52) | m;
}}
static uint32_t cls(uint64_t b) {{ return (uint32_t)(((b >> 63) << 11) | ((b >> 52) & 2047u)); }}
/* seen[p]: classes / class pairs, read back from the generated operand BITS. */
static uint8_t seen[{np}][{seen_len}];
static double {decl}ro[CHUNK], co[CHUNK], ro2[CHUNK], co2[CHUNK];
static uint64_t checked = 0, mism = 0, maxulp = 0, rnw = 0, cnw = 0, nfirst = 0;
static void flush(long long n) {{
    const int tier_kind = {tier_kind};
    const uint64_t tier_n = {tier_n};
    for (long long i = 0; i < n; ++i) {{
        ro[i] = f_of({sa}ull); co[i] = f_of({sa}ull);
        ro2[i] = f_of({sb}ull); co2[i] = f_of({sb}ull);
    }}
    {rname}({ins}ro, n);
    {cname}({ins}co, n);
    {rname}({ins}ro2, n);
    {cname}({ins}co2, n);
    for (long long i = 0; i < n; ++i) {{
        uint64_t rb = b_of(ro[i]), cb = b_of(co[i]);
        if (rb == {sa}ull && b_of(ro2[i]) == {sb}ull) ++rnw;
        if (cb == {sa}ull && b_of(co2[i]) == {sb}ull) ++cnw;
        int rn = ro[i] != ro[i], cn = co[i] != co[i];
        int bad;
        if (rn || cn) bad = (rn != cn);
        else if (tier_kind == 0) bad = (rb != cb);
        else {{
            int64_t a = key_of(rb), b = key_of(cb);
            uint64_t d = (a > b) ? (uint64_t)a - (uint64_t)b : (uint64_t)b - (uint64_t)a;
            if (d > maxulp) maxulp = d;
            bad = (d > tier_n);
        }}
        ++checked;
        if (bad) {{
            ++mism;
            if (nfirst < 5) {{
                printf("{first_fmt}\n", {first_args}(unsigned long long)rb, (unsigned long long)cb);
                ++nfirst;
            }}
        }}
    }}
}}
int main(void) {{
    long long n = 0;
    uint64_t cells = 0, edge_tuples = 0;
{body}    if (n) flush(n);
    printf("checked %llu\nmismatches %llu\nmax_ulp %llu\nref_never_wrote %llu\ncand_never_wrote %llu\n",
        (unsigned long long)checked, (unsigned long long)mism, (unsigned long long)maxulp,
        (unsigned long long)rnw, (unsigned long long)cnw);
    for (int p = 0; p < {np}; ++p) {{
        uint64_t c = 0;
        for (uint64_t i = 0; i < {seen_len}ull; ++i) c += seen[p][i];
        printf("covered %d %llu\n", p, (unsigned long long)c);
    }}
    printf("cells %llu\nedge_tuples %llu\n", (unsigned long long)cells, (unsigned long long)edge_tuples);
    return 0;
}}
"#,
        ref_src = reference.source,
        cand_src = cand.source,
        chunk = CHUNK,
        seed = spec.seed,
        edges = edge_list(),
        sa = SENTINEL_A,
        sb = SENTINEL_B,
        rname = reference.name,
        cname = cand.name,
    )
}

fn compare_sampled(
    work: &Path,
    tag: &str,
    arity: u32,
    reference: &Kernel,
    cand: &Kernel,
    tier: Tier,
    spec: SampleSpec,
) -> Result<SampledReport, String> {
    assert!(spec.per_cell >= 1);
    assert_ne!(
        reference.name, cand.name,
        "kernels must have distinct names"
    );
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    let c = work.join(format!("{tag}.c"));
    let exe = work.join(format!("{tag}.exe"));
    let _ = std::fs::remove_file(&exe);
    std::fs::write(&c, driver(reference, cand, arity, tier, spec)).map_err(|e| e.to_string())?;
    compile(&c, &exe)?;
    let out = Command::new(&exe)
        .output()
        .map_err(|e| format!("run: {e}"))?;
    if !out.status.success() {
        return Err(format!("harness exited {:?}", out.status.code()));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut r = Report {
        exhaustive: false,
        ..Report::default()
    };
    let np = if arity == 3 { 3 } else { 1 };
    let mut covered = vec![0u64; np];
    let mut first_failures = Vec::new();
    let (mut cells, mut edge_tuples, mut seen) = (0u64, 0u64, 0usize);
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let num = |v: &str| v.parse::<u64>().map_err(|e| format!("{line}: {e}"));
        match f.as_slice() {
            ["first", rest @ ..] if rest.len() == arity as usize + 2 => {
                let bits = rest.iter().map(|v| num(v)).collect::<Result<Vec<_>, _>>()?;
                let k = arity as usize;
                first_failures.push(Failure {
                    operands: bits[..k].to_vec(),
                    reference: bits[k],
                    candidate: bits[k + 1],
                });
            }
            ["checked", v] => (r.checked, seen) = (num(v)?, seen + 1),
            ["mismatches", v] => (r.mismatches, seen) = (num(v)?, seen + 1),
            ["max_ulp", v] => (r.max_ulp, seen) = (num(v)?, seen + 1),
            ["ref_never_wrote", v] => (r.ref_never_wrote, seen) = (num(v)?, seen + 1),
            ["cand_never_wrote", v] => (r.cand_never_wrote, seen) = (num(v)?, seen + 1),
            ["covered", p, v] => {
                let p = num(p)? as usize;
                if p >= np {
                    return Err(format!("{line}: operand pair out of range"));
                }
                (covered[p], seen) = (num(v)?, seen + 1)
            }
            ["cells", v] => (cells, seen) = (num(v)?, seen + 1),
            ["edge_tuples", v] => (edge_tuples, seen) = (num(v)?, seen + 1),
            _ => {}
        }
    }
    let want = 7 + np;
    if seen != want {
        return Err(format!(
            "harness output incomplete ({seen}/{want} summary lines):\n{text}"
        ));
    }
    let per_set = if arity == 1 {
        OPERAND_CLASSES
    } else {
        CLASS_PAIRS
    };
    if covered.iter().any(|&c| c != per_set) {
        return Err(format!(
            "class coverage {covered:?}, expected {per_set} for each set"
        ));
    }
    if r.checked != spec.tuples(arity)
        || cells != SampleSpec::cells(arity)
        || edge_tuples != SampleSpec::edge_tuples(arity)
    {
        return Err(format!(
            "checked {} tuples over {cells} cells + {edge_tuples} edge tuples, expected {} over {} + {}",
            r.checked,
            spec.tuples(arity),
            SampleSpec::cells(arity),
            SampleSpec::edge_tuples(arity)
        ));
    }
    Ok(SampledReport {
        report: r,
        arity,
        spec,
        cells,
        edge_tuples,
        covered,
        first_failures,
    })
}

/// Compare two unary f64 kernels over the stratified sample in the module docs. Never accepts.
pub fn compare_unary_f64_sampled(
    work: &Path,
    tag: &str,
    reference: &Kernel,
    cand: &Kernel,
    tier: Tier,
    spec: SampleSpec,
) -> Result<SampledReport, String> {
    compare_sampled(work, tag, 1, reference, cand, tier, spec)
}

/// Compare two binary f64 kernels over the stratified sample in the module docs. Never accepts.
pub fn compare_binary_f64_sampled(
    work: &Path,
    tag: &str,
    reference: &Kernel,
    cand: &Kernel,
    tier: Tier,
    spec: SampleSpec,
) -> Result<SampledReport, String> {
    compare_sampled(work, tag, 2, reference, cand, tier, spec)
}

/// Compare two ternary f64 kernels over the pairwise-stratified sample in the module docs.
/// Never accepts.
pub fn compare_ternary_f64_sampled(
    work: &Path,
    tag: &str,
    reference: &Kernel,
    cand: &Kernel,
    tier: Tier,
    spec: SampleSpec,
) -> Result<SampledReport, String> {
    compare_sampled(work, tag, 3, reference, cand, tier, spec)
}
