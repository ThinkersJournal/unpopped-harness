// SPDX-License-Identifier: MIT OR Apache-2.0
//! Binary f32 comparison by stratified sampling. A sample can never accept.
//!
//! Exhaustive testing of a binary f32 body would take 2^64 pairs, so this run is a screen, not
//! a proof: its verdict is [`ExactnessVerdict::Flushes`](unpopped::exactness::ExactnessVerdict::Flushes) (a difference was seen) or nothing at all
//! (see [`crate::exactness::verdict_of`]). `Report::exhaustive` is always `false` here, so
//! `Report::accepted` is always `false` too.
//!
//! The sample is stratified so that a defect confined to one class of operand pair is still hit.
//! An operand's class is its sign and biased exponent field: 2 x 256 = 512 classes, which include
//! the zeros, every subnormal, every normal binade, the infinities and the NaNs. Every ordered
//! pair of classes (512 x 512 = 262144 cells) receives [`SampleSpec::per_cell`] pairs, with the
//! mantissas drawn from a seeded generator and from the corner mantissas (all-zero, all-one, the
//! quiet bit). On top of that, a fixed grid of 16 edge values is crossed with itself. The sample
//! size and strata are returned with the report, so a verdict can say what it covered.
//!
//! A defect confined to a *mantissa* pattern inside one cell can still be missed; that is why a
//! clean sample concludes nothing.

use std::path::Path;
use std::process::Command;

use crate::{compile, Kernel, Report, Tier, CHUNK, SENTINEL_A, SENTINEL_B};

/// Operand classes: sign (2) x biased exponent field (256).
pub const OPERAND_CLASSES: u64 = 512;
/// Ordered pairs of operand classes.
pub const CELLS: u64 = OPERAND_CLASSES * OPERAND_CLASSES;
/// The fixed edge values, as bits; the grid crosses this list with itself.
pub const EDGE_VALUES: [u32; 16] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // smallest subnormal
    0x8000_0001,
    0x007f_ffff, // largest subnormal
    0x807f_ffff,
    0x0080_0000, // smallest normal
    0x3f80_0000, // 1
    0xbf80_0000, // -1
    0x7f7f_ffff, // largest finite
    0xff7f_ffff,
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // quiet NaN
    0x7fa0_0001, // signalling NaN
    0xffc0_0000, // negative quiet NaN
];

/// How big the sample is and where its randomness comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleSpec {
    /// Pairs drawn for each of the [`CELLS`] ordered class pairs.
    pub per_cell: u32,
    pub seed: u64,
}

impl Default for SampleSpec {
    fn default() -> Self {
        SampleSpec {
            per_cell: 4,
            seed: 0x005e_ed0f,
        }
    }
}

impl SampleSpec {
    /// Pairs a run with this spec compares.
    pub fn pairs(&self) -> u64 {
        CELLS * self.per_cell as u64 + (EDGE_VALUES.len() * EDGE_VALUES.len()) as u64
    }
}

/// A sampled run: the comparison plus exactly what it covered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampledReport {
    /// `exhaustive` is `false`, so this never accepts. `first` holds
    /// `(in0_bits, ref_bits, cand_bits)` for the first failing pairs.
    pub report: Report,
    pub spec: SampleSpec,
    /// Operand-class cells visited (always [`CELLS`] on success).
    pub cells: u64,
    /// Edge-grid pairs compared in addition to the cells.
    pub edge_pairs: u64,
    /// `(in0_bits, in1_bits)` of the first failing pairs, parallel to `report.first`.
    pub first_operands: Vec<(u32, u32)>,
}

impl Kernel {
    /// Lower a binary f32 elementwise `OpDef` through the cpu-c emitter. The exported symbol is
    /// `name(const float* in0, const float* in1, float* out, long long n)`.
    pub fn from_binary_f32(op: &unpopped::ir::OpDef) -> Kernel {
        use unpopped_vocab::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};
        // Alignment 4 defeats vectorization, giving the scalar schedule (as the cpu-c tests do).
        let d = OperandDesc::new(1, &[CHUNK as i64 - 1], &[1], ElementKind::F32, 4);
        let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
        let k = unpopped::generate(op, &key, &unpopped_cpu_c::CpuC);
        Kernel {
            name: k.name,
            source: k.source,
        }
    }
}

fn edge_list() -> String {
    EDGE_VALUES
        .iter()
        .map(|b| format!("0x{b:08x}u"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn driver(reference: &Kernel, cand: &Kernel, tier: Tier, spec: SampleSpec) -> String {
    let (tier_kind, tier_n) = match tier {
        Tier::Exact => (0, 0),
        Tier::Ulp(n) => (1, n),
    };
    format!(
        r#"{ref_src}
{cand_src}
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#define CHUNK {chunk}
static float f_of(uint32_t b) {{ float f; memcpy(&f, &b, 4); return f; }}
static uint32_t b_of(float f) {{ uint32_t b; memcpy(&b, &f, 4); return b; }}
static int64_t key_of(uint32_t b) {{
    return (b & 0x80000000u) ? -(int64_t)(b & 0x7fffffffu) : (int64_t)b;
}}
static uint64_t rng_state = {seed}ull;
static uint64_t rng(void) {{
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}}
static const uint32_t EDGE[16] = {{ {edges} }};
static const uint32_t CORNER[3] = {{ 0u, 0x7fffffu, 0x400000u }};
/* One operand of class c (sign<<8 | exponent field): a corner mantissa 3 times in 8, else random. */
static uint32_t operand(uint32_t c) {{
    uint64_t r = rng();
    uint32_t m = ((r & 7u) < 3u) ? CORNER[r & 7u] : (uint32_t)((r >> 8) & 0x7fffffu);
    return ((c >> 8) << 31) | ((c & 255u) << 23) | m;
}}
static float in0[CHUNK], in1[CHUNK], ro[CHUNK], co[CHUNK], ro2[CHUNK], co2[CHUNK];
static uint64_t checked = 0, mism = 0, maxulp = 0, rnw = 0, cnw = 0, nfirst = 0;
static void flush(long long n) {{
    const int tier_kind = {tier_kind};
    const uint64_t tier_n = {tier_n};
    for (long long i = 0; i < n; ++i) {{
        ro[i] = f_of({sa}u); co[i] = f_of({sa}u);
        ro2[i] = f_of({sb}u); co2[i] = f_of({sb}u);
    }}
    {rname}(in0, in1, ro, n);
    {cname}(in0, in1, co, n);
    {rname}(in0, in1, ro2, n);
    {cname}(in0, in1, co2, n);
    for (long long i = 0; i < n; ++i) {{
        uint32_t rb = b_of(ro[i]), cb = b_of(co[i]);
        if (rb == {sa}u && b_of(ro2[i]) == {sb}u) ++rnw;
        if (cb == {sa}u && b_of(co2[i]) == {sb}u) ++cnw;
        int rn = ro[i] != ro[i], cn = co[i] != co[i];
        int bad;
        if (rn || cn) bad = (rn != cn);
        else if (tier_kind == 0) bad = (rb != cb);
        else {{
            int64_t d = key_of(rb) - key_of(cb);
            if (d < 0) d = -d;
            if ((uint64_t)d > maxulp) maxulp = (uint64_t)d;
            bad = ((uint64_t)d > tier_n);
        }}
        ++checked;
        if (bad) {{
            ++mism;
            if (nfirst < 5) {{ printf("first %u %u %u %u\n", b_of(in0[i]), b_of(in1[i]), rb, cb); ++nfirst; }}
        }}
    }}
}}
int main(void) {{
    long long n = 0;
    uint64_t cells = 0, edge_pairs = 0;
    for (uint32_t ca = 0; ca < 512u; ++ca) {{
        for (uint32_t cb = 0; cb < 512u; ++cb) {{
            for (uint32_t k = 0; k < {per_cell}u; ++k) {{
                in0[n] = f_of(operand(ca)); in1[n] = f_of(operand(cb));
                if (++n == CHUNK) {{ flush(n); n = 0; }}
            }}
            ++cells;
        }}
    }}
    for (int a = 0; a < 16; ++a) for (int b = 0; b < 16; ++b) {{
        in0[n] = f_of(EDGE[a]); in1[n] = f_of(EDGE[b]); ++edge_pairs;
        if (++n == CHUNK) {{ flush(n); n = 0; }}
    }}
    if (n) flush(n);
    printf("checked %llu\nmismatches %llu\nmax_ulp %llu\nref_never_wrote %llu\ncand_never_wrote %llu\n",
        (unsigned long long)checked, (unsigned long long)mism, (unsigned long long)maxulp,
        (unsigned long long)rnw, (unsigned long long)cnw);
    printf("cells %llu\nedge_pairs %llu\n", (unsigned long long)cells, (unsigned long long)edge_pairs);
    return 0;
}}
"#,
        ref_src = reference.source,
        cand_src = cand.source,
        chunk = CHUNK,
        seed = spec.seed,
        edges = edge_list(),
        per_cell = spec.per_cell,
        tier_kind = tier_kind,
        tier_n = tier_n,
        sa = SENTINEL_A,
        sb = SENTINEL_B,
        rname = reference.name,
        cname = cand.name,
    )
}

/// Compare two binary f32 kernels over the stratified sample described in the module docs.
/// The returned report never accepts.
pub fn compare_binary_f32_sampled(
    work: &Path,
    tag: &str,
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
    std::fs::write(&c, driver(reference, cand, tier, spec)).map_err(|e| e.to_string())?;
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
    let mut first_operands = Vec::new();
    let (mut cells, mut edge_pairs, mut seen) = (0u64, 0u64, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let num = |v: &str| v.parse::<u64>().map_err(|e| format!("{line}: {e}"));
        match f.as_slice() {
            ["first", a, b, rb, cb] => {
                let (a, b) = (num(a)? as u32, num(b)? as u32);
                r.first.push((a, num(rb)? as u32, num(cb)? as u32));
                first_operands.push((a, b));
            }
            ["checked", v] => (r.checked, seen) = (num(v)?, seen + 1),
            ["mismatches", v] => (r.mismatches, seen) = (num(v)?, seen + 1),
            ["max_ulp", v] => (r.max_ulp, seen) = (num(v)?, seen + 1),
            ["ref_never_wrote", v] => (r.ref_never_wrote, seen) = (num(v)?, seen + 1),
            ["cand_never_wrote", v] => (r.cand_never_wrote, seen) = (num(v)?, seen + 1),
            ["cells", v] => (cells, seen) = (num(v)?, seen + 1),
            ["edge_pairs", v] => (edge_pairs, seen) = (num(v)?, seen + 1),
            _ => {}
        }
    }
    if seen != 7 {
        return Err(format!(
            "harness output incomplete ({seen}/7 summary lines):\n{text}"
        ));
    }
    if r.checked != spec.pairs() || cells != CELLS {
        return Err(format!(
            "checked {} pairs over {cells} cells, expected {} over {CELLS}",
            r.checked,
            spec.pairs()
        ));
    }
    Ok(SampledReport {
        report: r,
        spec,
        cells,
        edge_pairs,
        first_operands,
    })
}
