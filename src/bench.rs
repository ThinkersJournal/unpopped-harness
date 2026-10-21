// SPDX-License-Identifier: MIT OR Apache-2.0
//! Track 4: per-op microkernels with normalized timings (cpu-c only).
//!
//! One C program holds every unary op's emitted kernel plus a plain copy loop.
//! Each kernel is timed over an L2-resident chunk (min of REPS), then reported as
//! ns/element and as a ratio to the copy loop on the same machine and run, so the
//! number is comparable across ops here and its shape across targets later.

use std::path::Path;
use std::process::Command;

use unpopped::ir::{input, OpDef, UnaryOp};
use unpopped_vocab::ElementKind;

use crate::{compile, Kernel, CHUNK};

pub const UNARY_OPS: &[(&str, UnaryOp)] = &[
    ("neg", UnaryOp::Neg),
    ("abs", UnaryOp::Abs),
    ("sqr", UnaryOp::Sqr),
    ("sqrt", UnaryOp::Sqrt),
    ("rsqrt", UnaryOp::Rsqrt),
    ("recip", UnaryOp::Recip),
    ("exp", UnaryOp::Exp),
    ("log", UnaryOp::Log),
    ("tanh", UnaryOp::Tanh),
    ("sigmoid", UnaryOp::Sigmoid),
    ("relu", UnaryOp::Relu),
    ("erf", UnaryOp::Erf),
    ("gelu", UnaryOp::Gelu),
    ("silu", UnaryOp::Silu),
    ("sin", UnaryOp::Sin),
    ("cos", UnaryOp::Cos),
    ("floor", UnaryOp::Floor),
    ("ceil", UnaryOp::Ceil),
    ("round", UnaryOp::Round),
    ("sign", UnaryOp::Sign),
    ("step", UnaryOp::Step),
    ("erfc", UnaryOp::Erfc),
    ("trunc", UnaryOp::Trunc),
    ("exp2", UnaryOp::Exp2),
    ("expm1", UnaryOp::Expm1),
    ("log2", UnaryOp::Log2),
    ("log10", UnaryOp::Log10),
    ("log1p", UnaryOp::Log1p),
    ("sinh", UnaryOp::Sinh),
    ("cosh", UnaryOp::Cosh),
    ("tan", UnaryOp::Tan),
    ("asin", UnaryOp::Asin),
    ("acos", UnaryOp::Acos),
    ("atan", UnaryOp::Atan),
    ("asinh", UnaryOp::Asinh),
    ("acosh", UnaryOp::Acosh),
    ("atanh", UnaryOp::Atanh),
    ("cbrt", UnaryOp::Cbrt),
    ("lgamma", UnaryOp::Lgamma),
];

#[derive(Clone, Debug)]
pub struct Timing {
    pub op: String,
    pub ns_per_elem: f64,
    /// ns/element divided by the copy loop's ns/element in the same run.
    pub vs_copy: f64,
}

pub fn time_unary_ops(work: &Path, reps: u32, calls: u32) -> Result<Vec<Timing>, String> {
    let mut srcs = String::new();
    let mut calls_c = String::new();
    for (name, op) in UNARY_OPS {
        let def = OpDef::elementwise(
            &format!("def_{name}"),
            1,
            &[ElementKind::F32],
            input(0).unary(*op),
        );
        let mut k = Kernel::from_unary_f32(&def);
        // Unique, C-safe symbol per op regardless of what the emitter names it.
        let sym = format!("k_{name}");
        srcs.push_str(&k.source.replace(&k.name, &sym));
        k.name = sym.clone();
        calls_c.push_str(&format!(
            "    t = bench({sym}); printf(\"{name} %.6f\\n\", t / ((double)n_calls * CHUNK));\n"
        ));
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
        srcs = srcs,
        chunk = CHUNK,
        reps = reps,
        calls = calls,
        calls_c = calls_c,
    );
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    let c = work.join("bench.c");
    let exe = work.join("bench.exe");
    let _ = std::fs::remove_file(&exe);
    std::fs::write(&c, prog).map_err(|e| e.to_string())?;
    compile(&c, &exe)?;
    let out = Command::new(&exe)
        .output()
        .map_err(|e| format!("run: {e}"))?;
    if !out.status.success() {
        return Err(format!("bench exited {:?}", out.status.code()));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut copy = None;
    let mut raw = Vec::new();
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
    let copy = copy.ok_or("no copy baseline in output")?;
    if raw.len() != UNARY_OPS.len() {
        return Err(format!(
            "got {} op timings, expected {}:\n{text}",
            raw.len(),
            UNARY_OPS.len()
        ));
    }
    Ok(raw
        .into_iter()
        .map(|(op, ns)| Timing {
            op,
            ns_per_elem: ns,
            vs_copy: ns / copy,
        })
        .collect())
}
