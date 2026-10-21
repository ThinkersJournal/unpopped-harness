// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cross-target, fused chains: the same `d`-deep unary chains timed on cpu-c and on CUDA.
//!
//! Single-op kernels are bandwidth-bound on the GPU (all 39 = 1.00x copy), so the targets cannot be
//! told apart. A chain raises arithmetic intensity per element until the GPU leaves the memory
//! roofline; the depth at which each chain's ratio departs from 1.00x is the measurement.

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};
use baracuda_nvrtc::Program;
use unpopped::ir::{input, Expr, OpDef, UnaryOp};
use unpopped_vocab::ElementKind;

use crate::{compile, Kernel, CHUNK};

pub const DEPTHS: &[usize] = &[1, 2, 4, 8, 16, 32];
pub const N: usize = 1 << 24;

/// (family name, op sequence cycled to depth d)
pub const FAMILIES: &[(&str, &[UnaryOp])] = &[
    ("tanh", &[UnaryOp::Tanh]),
    ("sqrt_add", &[UnaryOp::Sqrt, UnaryOp::Sqr, UnaryOp::Sqrt]),
    (
        "mixed",
        &[UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Tanh, UnaryOp::Sigmoid],
    ),
];

fn chain_def(ops: &[UnaryOp], d: usize) -> OpDef {
    let mut e: Expr = input(0);
    for i in 0..d {
        e = e.unary(ops[i % ops.len()]);
    }
    OpDef::elementwise("chain", 1, &[ElementKind::F32], e)
}

pub fn chain_names() -> Vec<String> {
    let mut v = Vec::new();
    for (f, _) in FAMILIES {
        for d in DEPTHS {
            v.push(format!("{f}_d{d}"));
        }
    }
    v
}

#[derive(Clone, Debug)]
pub struct ChainTiming {
    pub name: String,
    pub ns_per_elem: f64,
    pub vs_copy: f64,
}

pub fn time_chains_cpu(work: &Path, reps: u32, calls: u32) -> Result<Vec<ChainTiming>, String> {
    let (mut srcs, mut calls_c) = (String::new(), String::new());
    for (fam, ops) in FAMILIES {
        for d in DEPTHS {
            let name = format!("{fam}_d{d}");
            let k = Kernel::from_unary_f32(&chain_def(ops, *d));
            let sym = format!("k_{name}");
            srcs.push_str(&k.source.replace(&k.name, &sym));
            calls_c.push_str(&format!(
                "    t = bench({sym}); printf(\"{name} %.6f\\n\", t / ((double)n_calls * CHUNK));\n"
            ));
        }
    }
    let prog = format!(
        r#"{srcs}
#include <stdio.h>
#include <stdint.h>
#include <windows.h>
#define CHUNK {chunk}
#define REPS {reps}
#define CALLS {calls}
static float in0[CHUNK], out0[CHUNK];
static const long long n_calls = CALLS;
static double now(void) {{ LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart * 1e9 / (double)f.QuadPart; }}
static void k_copy(const float* a, float* o, long long n) {{ for (long long i = 0; i < n; ++i) o[i] = a[i]; }}
static double bench(void (*f)(const float*, float*, long long)) {{
    double best = 1e300;
    for (int r = 0; r < REPS; ++r) {{
        double t0 = now();
        for (int c = 0; c < CALLS; ++c) {{ f(in0, out0, CHUNK); }}
        double dt = now() - t0;
        if (dt < best) best = dt;
    }}
    return best;
}}
int main(void) {{
    for (int i = 0; i < CHUNK; ++i) in0[i] = 0.25f + (float)(i % 1024) / 1024.0f;
    double t;
    t = bench(k_copy); printf("copy %.6f\n", t / ((double)n_calls * CHUNK));
{calls_c}    printf("sink %f\n", (double)out0[17]);
    return 0;
}}
"#,
        chunk = CHUNK,
    );
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    let (c, exe) = (work.join("chain.c"), work.join("chain.exe"));
    let _ = std::fs::remove_file(&exe);
    std::fs::write(&c, prog).map_err(|e| e.to_string())?;
    compile(&c, &exe)?;
    let out = Command::new(&exe)
        .output()
        .map_err(|e| format!("run: {e}"))?;
    if !out.status.success() {
        return Err(format!("chain exited {:?}", out.status.code()));
    }
    parse(&String::from_utf8_lossy(&out.stdout))
}

fn parse(text: &str) -> Result<Vec<ChainTiming>, String> {
    let (mut copy, mut raw) = (None, Vec::new());
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if let [name, v] = f.as_slice() {
            if *name == "sink" {
                continue;
            }
            let v: f64 = v.parse().map_err(|_| format!("bad line {line}"))?;
            if *name == "copy" {
                copy = Some(v);
            } else {
                raw.push((name.to_string(), v));
            }
        }
    }
    let copy = copy.ok_or("no copy baseline")?;
    if raw.len() != chain_names().len() {
        return Err(format!(
            "got {} timings, expected {}:\n{text}",
            raw.len(),
            chain_names().len()
        ));
    }
    Ok(raw
        .into_iter()
        .map(|(name, ns)| ChainTiming {
            name,
            ns_per_elem: ns,
            vs_copy: ns / copy,
        })
        .collect())
}

const COPY_SRC: &str = r#"extern "C" __global__ void k_copy(const float* a, float* o, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i];
}"#;

/// Returns (device name, timings).
pub fn time_chains_gpu(reps: u32, calls: u32) -> Result<(String, Vec<ChainTiming>), String> {
    baracuda_driver::init().map_err(|e| format!("cuda init: {e:?}"))?;
    let device = Device::get(0).map_err(|e| format!("device: {e:?}"))?;
    let dev_name = device.name().map_err(|e| format!("name: {e:?}"))?;
    let (maj, min) = device
        .compute_capability()
        .map_err(|e| format!("cc: {e:?}"))?;
    let arch = format!("--gpu-architecture=compute_{maj}{min}");
    let ctx = Context::new(&device).map_err(|e| format!("ctx: {e:?}"))?;
    let stream = Stream::new(&ctx).map_err(|e| format!("stream: {e:?}"))?;
    let d_in = DeviceBuffer::<f32>::new(&ctx, N).map_err(|e| format!("alloc: {e:?}"))?;
    let d_out = DeviceBuffer::<f32>::new(&ctx, N).map_err(|e| format!("alloc: {e:?}"))?;
    let host: Vec<f32> = (0..N).map(|i| 0.25 + (i % 1024) as f32 / 1024.0).collect();
    d_in.copy_from_host(&host)
        .map_err(|e| format!("h2d: {e:?}"))?;

    let time = |src: &str, name: &str| -> Result<f64, String> {
        let ptx = Program::compile(src, "chain.cu", &[&arch])
            .map_err(|e| format!("nvrtc {name}: {e:?}"))?;
        let module = Module::load_ptx(&ctx, &ptx).map_err(|e| format!("module {name}: {e:?}"))?;
        let f = module
            .get_function(name)
            .map_err(|e| format!("function {name}: {e:?}"))?;
        let (ip, op_, nn) = (d_in.as_raw(), d_out.as_raw(), N as i64);
        let grid = (N as u32).div_ceil(256);
        let launch = || -> Result<(), String> {
            unsafe {
                f.launch()
                    .grid((grid, 1, 1))
                    .block((256, 1, 1))
                    .stream(&stream)
                    .arg(&ip)
                    .arg(&op_)
                    .arg(&nn)
                    .launch()
                    .map_err(|e| format!("launch {name}: {e:?}"))
            }
        };
        d_out
            .copy_from_host(&vec![f32::from_bits(0x7fc0_dead); N])
            .map_err(|e| format!("h2d: {e:?}"))?;
        launch()?;
        stream.synchronize().map_err(|e| format!("sync: {e:?}"))?;
        let mut chk = vec![0f32; N];
        d_out
            .copy_to_host(&mut chk)
            .map_err(|e| format!("d2h: {e:?}"))?;
        let unwritten = chk.iter().filter(|v| v.to_bits() == 0x7fc0_dead).count();
        if unwritten != 0 {
            return Err(format!(
                "{name}: {unwritten} of {N} outputs still hold the sentinel"
            ));
        }
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t0 = Instant::now();
            for _ in 0..calls {
                launch()?;
            }
            stream.synchronize().map_err(|e| format!("sync: {e:?}"))?;
            best = best.min(t0.elapsed().as_nanos() as f64);
        }
        Ok(best / (calls as f64 * N as f64))
    };

    let copy = time(COPY_SRC, "k_copy")?;
    let mut out = Vec::new();
    for (fam, ops) in FAMILIES {
        for d in DEPTHS {
            let k = Kernel::from_unary_f32_cuda(&chain_def(ops, *d));
            let ns = time(&k.source, &k.name)?;
            out.push(ChainTiming {
                name: format!("{fam}_d{d}"),
                ns_per_elem: ns,
                vs_copy: ns / copy,
            });
        }
    }
    println!("copy {copy:.6} ns/elem (N={N})");
    Ok((dev_name, out))
}
