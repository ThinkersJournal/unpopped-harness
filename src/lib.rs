// SPDX-License-Identifier: MIT OR Apache-2.0
//! Track 1: the contract and the harness.
//!
//! A reference kernel is the spec. A candidate is accepted only if it meets the
//! [`Tier`] against the reference on EVERY f32 input (stride 1) — exhaustive,
//! so there is no sample for a search to overfit. A stride > 1 is a screening
//! mode that cannot accept; see [`Report::accepted`].

pub mod bench;
pub mod chain_bench;
pub mod gpu;
pub mod gpu_bench;

use std::path::{Path, PathBuf};
use std::process::Command;

use unpopped::ir::OpDef;
use unpopped::{generate, GeneratedKernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

/// What "correct" means for a candidate against its reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Bit-identical, except that any NaN equals any NaN (payloads and the sign of
    /// a NaN are not compared). `-0.0` and `+0.0` are DIFFERENT.
    Exact,
    /// Within `n` ULP of the reference in the f32 total ordering. `-0.0` and
    /// `+0.0` are the same point. A NaN must meet a NaN.
    Ulp(u32),
}

/// A kernel as C source exporting `name(const float* in0, float* out, long long n)`.
#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: String,
    pub source: String,
}

/// Elements per kernel call. The key shape and the C loop agree on it.
pub const CHUNK: usize = 1 << 16;

impl Kernel {
    /// Lower a unary f32 elementwise `OpDef` through the cpu-c emitter.
    pub fn from_unary_f32(op: &OpDef) -> Kernel {
        let d = OperandDesc::new(1, &[CHUNK as i64 - 1], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[d, d], ArchSku::Sm89);
        let k: GeneratedKernel = generate(op, &key, &CpuC);
        Kernel {
            name: k.name,
            source: k.source,
        }
    }
}

impl Kernel {
    /// Lower a unary f32 elementwise `OpDef` through the CUDA emitter (.cu source).
    pub fn from_unary_f32_cuda(op: &OpDef) -> Kernel {
        let d = OperandDesc::new(1, &[CHUNK as i64 - 1], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[d, d], ArchSku::Sm89);
        let k: GeneratedKernel = generate(op, &key, &baracuda_cuda_emit::Cuda);
        Kernel {
            name: k.name,
            source: k.source,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Inputs compared.
    pub checked: u64,
    /// Inputs where the candidate broke the contract.
    pub mismatches: u64,
    /// Largest ULP distance seen among non-NaN pairs (Ulp tier only; 0 for Exact).
    pub max_ulp: u64,
    /// Outputs still carrying the harness sentinel: the kernel never wrote them.
    pub ref_never_wrote: u64,
    pub cand_never_wrote: u64,
    /// First few failing inputs as `(input_bits, ref_bits, cand_bits)`.
    pub first: Vec<(u32, u32, u32)>,
    /// True iff every one of the 2^32 inputs was compared (stride 1).
    pub exhaustive: bool,
}

impl Report {
    /// The only way to accept a candidate. Requires exhaustive coverage, zero
    /// mismatches, and that neither kernel left a cell unwritten.
    pub fn accepted(&self) -> bool {
        self.exhaustive
            && self.mismatches == 0
            && self.ref_never_wrote == 0
            && self.cand_never_wrote == 0
    }
}

fn vcvars() -> PathBuf {
    std::env::var_os("UNPOPPED_VCVARS").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(
            r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
        )
    })
}

/// Compile with `cl /O2` through a generated .bat (one quoting layer).
pub fn compile(src: &Path, exe: &Path) -> Result<(), String> {
    compile_with(src, exe, "")
}

/// As [`compile`], with extra `cl` flags (e.g. `/LD` for a DLL).
pub fn compile_with(src: &Path, exe: &Path, flags: &str) -> Result<(), String> {
    let bat = src.with_extension("build.bat");
    let script = format!(
        "@echo off\r\ncall \"{}\" >nul 2>&1\r\ncl /nologo /O2 {} \"{}\" /Fe:\"{}\"\r\n",
        vcvars().display(),
        flags,
        src.display(),
        exe.display()
    );
    std::fs::write(&bat, script).map_err(|e| format!("write build script: {e}"))?;
    let out = Command::new(&bat)
        .current_dir(src.parent().unwrap_or(Path::new(".")))
        .output()
        .map_err(|e| format!("spawn: {e}"))?;
    if out.status.success() && exe.exists() {
        return Ok(());
    }
    Err(format!(
        "compile failed\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    ))
}

/// Two distinct NaN patterns. A cell counts as never-written only if it still
/// holds the sentinel after BOTH passes, each seeded with a different one: a
/// kernel that quiets a signalling NaN can legitimately emit 0x7fc0dead, but no
/// deterministic kernel emits two different values for one input.
pub(crate) const SENTINEL_A: u32 = 0x7fc0_dead;
pub(crate) const SENTINEL_B: u32 = 0xffc0_beef;

fn driver(reference: &Kernel, cand: &Kernel, tier: Tier, stride: u32) -> String {
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
static float in0[CHUNK], ro[CHUNK], co[CHUNK], ro2[CHUNK], co2[CHUNK];
int main(void) {{
    const uint64_t stride = {stride};
    const int tier_kind = {tier_kind};
    const uint64_t tier_n = {tier_n};
    uint64_t checked = 0, mism = 0, maxulp = 0, rnw = 0, cnw = 0, nfirst = 0;
    uint64_t pos = 0;
    while (pos < 4294967296ull) {{
        long long n = 0;
        for (; n < CHUNK && pos < 4294967296ull; ++n, pos += stride) in0[n] = f_of((uint32_t)pos);
        for (long long i = 0; i < n; ++i) {{
            ro[i] = f_of({sa}u); co[i] = f_of({sa}u);
            ro2[i] = f_of({sb}u); co2[i] = f_of({sb}u);
        }}
        {rname}(in0, ro, n);
        {cname}(in0, co, n);
        {rname}(in0, ro2, n);
        {cname}(in0, co2, n);
        for (long long i = 0; i < n; ++i) {{
            uint32_t ib = b_of(in0[i]), rb = b_of(ro[i]), cb = b_of(co[i]);
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
                if (nfirst < 5) {{ printf("first %u %u %u\n", ib, rb, cb); ++nfirst; }}
            }}
        }}
    }}
    printf("checked %llu\nmismatches %llu\nmax_ulp %llu\nref_never_wrote %llu\ncand_never_wrote %llu\n",
        (unsigned long long)checked, (unsigned long long)mism, (unsigned long long)maxulp,
        (unsigned long long)rnw, (unsigned long long)cnw);
    return 0;
}}
"#,
        ref_src = reference.source,
        cand_src = cand.source,
        chunk = CHUNK,
        sa = SENTINEL_A,
        sb = SENTINEL_B,
        rname = reference.name,
        cname = cand.name,
    )
}

/// Compare `cand` to `reference` over the f32 inputs whose bit patterns are
/// `0, stride, 2*stride, ...`. `stride == 1` is exhaustive.
pub fn compare_unary_f32(
    work: &Path,
    tag: &str,
    reference: &Kernel,
    cand: &Kernel,
    tier: Tier,
    stride: u32,
) -> Result<Report, String> {
    assert!(stride >= 1);
    assert_ne!(
        reference.name, cand.name,
        "kernels must have distinct names"
    );
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    let c = work.join(format!("{tag}.c"));
    let exe = work.join(format!("{tag}.exe"));
    let _ = std::fs::remove_file(&exe);
    std::fs::write(&c, driver(reference, cand, tier, stride)).map_err(|e| e.to_string())?;
    compile(&c, &exe)?;
    let out = Command::new(&exe)
        .output()
        .map_err(|e| format!("run: {e}"))?;
    if !out.status.success() {
        return Err(format!("harness exited {:?}", out.status.code()));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut r = Report {
        exhaustive: stride == 1,
        ..Report::default()
    };
    let mut seen = 0;
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["first", a, b, c] => {
                r.first
                    .push((a.parse().unwrap(), b.parse().unwrap(), c.parse().unwrap()))
            }
            ["checked", v] => {
                r.checked = v.parse().unwrap();
                seen += 1
            }
            ["mismatches", v] => {
                r.mismatches = v.parse().unwrap();
                seen += 1
            }
            ["max_ulp", v] => {
                r.max_ulp = v.parse().unwrap();
                seen += 1
            }
            ["ref_never_wrote", v] => {
                r.ref_never_wrote = v.parse().unwrap();
                seen += 1
            }
            ["cand_never_wrote", v] => {
                r.cand_never_wrote = v.parse().unwrap();
                seen += 1
            }
            _ => {}
        }
    }
    // A harness that printed less than it should is not a clean run.
    if seen != 5 {
        return Err(format!(
            "harness output incomplete ({seen}/5 summary lines):\n{text}"
        ));
    }
    let expect = (1u64 << 32).div_ceil(stride as u64);
    if r.checked != expect {
        return Err(format!("checked {} inputs, expected {expect}", r.checked));
    }
    Ok(r)
}
