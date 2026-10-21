// SPDX-License-Identifier: MIT OR Apache-2.0
//! Second target: a CUDA candidate checked against a cpu-c reference over every f32.
//!
//! The reference is the cpu-c kernel built as a DLL; the candidate runs on the GPU through
//! NVRTC and the driver. Same contract, same [`Report`], same two-sentinel never-wrote test
//! as the cpu-vs-cpu harness. The tier rules are re-implemented here in Rust, so
//! `tier_ok` has its own unit tests.

use std::path::Path;

use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};
use baracuda_nvrtc::Program;

use crate::{compile_with, Kernel, Report, Tier, SENTINEL_A, SENTINEL_B};

/// Elements per GPU launch.
/// Elements per GPU launch.
pub(crate) const GPU_CHUNK: usize = 1 << 22;

fn key_of(b: u32) -> i64 {
    if b & 0x8000_0000 != 0 {
        -((b & 0x7fff_ffff) as i64)
    } else {
        b as i64
    }
}

/// `Some(distance)` when the pair meets the tier; `None` when it breaks it.
/// Distance is 0 for NaN pairs and for the Exact tier.
pub fn tier_ok(tier: Tier, r: u32, c: u32) -> Option<u64> {
    let (rf, cf) = (f32::from_bits(r), f32::from_bits(c));
    if rf.is_nan() || cf.is_nan() {
        return if rf.is_nan() && cf.is_nan() {
            Some(0)
        } else {
            None
        };
    }
    match tier {
        Tier::Exact => {
            if r == c {
                Some(0)
            } else {
                None
            }
        }
        Tier::Ulp(n) => {
            let d = (key_of(r) - key_of(c)).unsigned_abs();
            if d <= n as u64 {
                Some(d)
            } else {
                None
            }
        }
    }
}

/// Build the cpu-c reference as a DLL exporting `ref_entry`.
fn build_ref_dll(work: &Path, tag: &str, reference: &Kernel) -> Result<std::path::PathBuf, String> {
    let c = work.join(format!("{tag}_ref.c"));
    let dll = work.join(format!("{tag}_ref.dll"));
    let _ = std::fs::remove_file(&dll);
    let src = format!(
        "{}\n__declspec(dllexport) void ref_entry(const float* in0, float* out, long long n) {{ {}(in0, out, n); }}\n",
        reference.source, reference.name
    );
    std::fs::write(&c, src).map_err(|e| e.to_string())?;
    compile_with(&c, &dll, "/LD")?;
    Ok(dll)
}

type RefFn = unsafe extern "C" fn(*const f32, *mut f32, i64);

/// Compare a CUDA `cand` to a cpu-c `reference` over bit patterns `0, stride, 2*stride, ...`.
/// `stride == 1` is exhaustive. `cand.source` must be CUDA C; `reference.source` plain C.
pub fn compare_unary_f32_gpu(
    work: &Path,
    tag: &str,
    reference: &Kernel,
    cand: &Kernel,
    tier: Tier,
    stride: u32,
) -> Result<Report, String> {
    baracuda_driver::init().map_err(|e| format!("cuda init: {e:?}"))?;
    let device = Device::get(0).map_err(|e| format!("device: {e:?}"))?;
    let (maj, min) = device
        .compute_capability()
        .map_err(|e| format!("cc: {e:?}"))?;
    let arch = format!("--gpu-architecture=compute_{maj}{min}");
    let ptx =
        Program::compile(&cand.source, "cand.cu", &[&arch]).map_err(|e| format!("nvrtc: {e:?}"))?;
    let ctx = Context::new(&device).map_err(|e| format!("ctx: {e:?}"))?;
    let stream = Stream::new(&ctx).map_err(|e| format!("stream: {e:?}"))?;
    let module = Module::load_ptx(&ctx, &ptx).map_err(|e| format!("module: {e:?}"))?;
    let kernel = module
        .get_function(&cand.name)
        .map_err(|e| format!("function: {e:?}"))?;

    let d_in = DeviceBuffer::<f32>::new(&ctx, GPU_CHUNK).map_err(|e| format!("alloc: {e:?}"))?;
    let d_out = DeviceBuffer::<f32>::new(&ctx, GPU_CHUNK).map_err(|e| format!("alloc: {e:?}"))?;

    let run_gpu = |input: &[f32], seed: f32, out: &mut [f32]| -> Result<(), String> {
        let n = input.len();
        // The device buffer is fixed-size; the tail beyond `n` is never read (kernel gets `n`).
        let mut padded = vec![0f32; GPU_CHUNK];
        padded[..n].copy_from_slice(input);
        d_in.copy_from_host(&padded)
            .map_err(|e| format!("h2d: {e:?}"))?;
        d_out
            .copy_from_host(&vec![seed; GPU_CHUNK])
            .map_err(|e| format!("h2d: {e:?}"))?;
        let (ip, op_, nn) = (d_in.as_raw(), d_out.as_raw(), n as i64);
        let block = 256u32;
        let grid = (n as u32).div_ceil(block);
        unsafe {
            kernel
                .launch()
                .grid((grid, 1, 1))
                .block((block, 1, 1))
                .stream(&stream)
                .arg(&ip)
                .arg(&op_)
                .arg(&nn)
                .launch()
                .map_err(|e| format!("launch: {e:?}"))?;
        }
        stream.synchronize().map_err(|e| format!("sync: {e:?}"))?;
        let mut full = vec![0f32; GPU_CHUNK];
        d_out
            .copy_to_host(&mut full)
            .map_err(|e| format!("d2h: {e:?}"))?;
        out[..n].copy_from_slice(&full[..n]);
        Ok(())
    };

    compare_with_runner(work, tag, reference, tier, stride, |_| false, run_gpu)
}

/// The comparison loop, shared by every GPU target: `run(input, seed, out)` runs the candidate over
/// `input` with its output cells pre-seeded with `seed`, and fills `out[..input.len()]`.
/// `waive(input_bits)` marks inputs whose mismatch is a declared, counted exception.
pub(crate) fn compare_with_runner(
    work: &Path,
    tag: &str,
    reference: &Kernel,
    tier: Tier,
    stride: u32,
    waive: fn(u32) -> bool,
    mut run: impl FnMut(&[f32], f32, &mut [f32]) -> Result<(), String>,
) -> Result<Report, String> {
    assert!(stride >= 1);
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    let dll = build_ref_dll(work, tag, reference)?;
    let lib =
        unsafe { libloading::Library::new(&dll) }.map_err(|e| format!("load ref dll: {e}"))?;
    let ref_fn: libloading::Symbol<RefFn> =
        unsafe { lib.get(b"ref_entry") }.map_err(|e| format!("ref_entry: {e}"))?;

    let sa = f32::from_bits(SENTINEL_A);
    let sb = f32::from_bits(SENTINEL_B);
    let mut rep = Report {
        exhaustive: stride == 1,
        ..Report::default()
    };
    let mut inp = vec![0f32; GPU_CHUNK];
    let mut ro = vec![0f32; GPU_CHUNK];
    let mut ro2 = vec![0f32; GPU_CHUNK];
    let mut co = vec![0f32; GPU_CHUNK];
    let mut co2 = vec![0f32; GPU_CHUNK];

    let mut pos: u64 = 0;
    while pos < (1u64 << 32) {
        let mut n = 0usize;
        while n < GPU_CHUNK && pos < (1u64 << 32) {
            inp[n] = f32::from_bits(pos as u32);
            n += 1;
            pos += stride as u64;
        }
        ro[..n].fill(sa);
        ro2[..n].fill(sb);
        unsafe {
            ref_fn(inp.as_ptr(), ro.as_mut_ptr(), n as i64);
            ref_fn(inp.as_ptr(), ro2.as_mut_ptr(), n as i64);
        }
        run(&inp[..n], sa, &mut co)?;
        run(&inp[..n], sb, &mut co2)?;
        for i in 0..n {
            let (ib, rb, cb) = (inp[i].to_bits(), ro[i].to_bits(), co[i].to_bits());
            if rb == SENTINEL_A && ro2[i].to_bits() == SENTINEL_B {
                rep.ref_never_wrote += 1;
            }
            if cb == SENTINEL_A && co2[i].to_bits() == SENTINEL_B {
                rep.cand_never_wrote += 1;
            }
            rep.checked += 1;
            match tier_ok(tier, rb, cb) {
                Some(d) => rep.max_ulp = rep.max_ulp.max(d),
                None if waive(ib) => rep.waived += 1,
                None => {
                    rep.mismatches += 1;
                    if rep.first.len() < 5 {
                        rep.first.push((ib, rb, cb));
                    }
                }
            }
        }
    }
    let expect = (1u64 << 32).div_ceil(stride as u64);
    if rep.checked != expect {
        return Err(format!("checked {} inputs, expected {expect}", rep.checked));
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_rules() {
        let nan1 = 0x7fc0_0000;
        let nan2 = 0xffc0_beef;
        assert_eq!(tier_ok(Tier::Exact, nan1, nan2), Some(0));
        assert!(tier_ok(Tier::Exact, nan1, 0).is_none());
        assert!(tier_ok(Tier::Exact, 0x8000_0000, 0).is_none()); // -0 != +0 under Exact
        assert_eq!(tier_ok(Tier::Ulp(1), 0x8000_0000, 0), Some(0)); // same point under Ulp
        assert_eq!(tier_ok(Tier::Ulp(1), 0x3f80_0000, 0x3f80_0001), Some(1));
        assert!(tier_ok(Tier::Ulp(1), 0x3f80_0000, 0x3f80_0002).is_none());
        assert!(tier_ok(Tier::Ulp(1), 0x8000_0001, 0x0000_0001).is_none()); // 2 apart across zero
    }
}
