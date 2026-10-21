// SPDX-License-Identifier: MIT OR Apache-2.0
//! Track 4, second target: the same 39 unary ops timed on the GPU (CUDA emitter via NVRTC).
//!
//! Device-resident buffers (no transfers), `N` elements per launch, min over REPS of CALLS
//! back-to-back launches, wall-clock around a stream sync. Ratio is to a hand-written copy
//! kernel on the same device and run, mirroring the cpu-c ratio in `bench.rs`.

use std::time::Instant;

use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};
use baracuda_nvrtc::Program;
use unpopped::ir::{input, OpDef};
use unpopped_vocab::ElementKind;

use crate::bench::{Timing, UNARY_OPS};
use crate::Kernel;

pub const N: usize = 1 << 24;

const COPY_SRC: &str = r#"extern "C" __global__ void k_copy(const float* a, float* o, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i];
}"#;

const HEAVY_SRC: &str = r#"extern "C" __global__ void k_heavy(const float* a, float* o, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = a[i]; for (int k = 0; k < 64; ++k) v = tanhf(v) + 0.5f; o[i] = v; }
}"#;

pub fn time_unary_ops_gpu(reps: u32, calls: u32) -> Result<Vec<Timing>, String> {
    baracuda_driver::init().map_err(|e| format!("cuda init: {e:?}"))?;
    let device = Device::get(0).map_err(|e| format!("device: {e:?}"))?;
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
        let ptx = Program::compile(src, "bench.cu", &[&arch])
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
        launch()?; // warm-up (lazy module init)
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
    let heavy = time(HEAVY_SRC, "k_heavy")?;
    println!(
        "control k_heavy {heavy:.6} ns/elem = {:.2}x copy (must be > 1)",
        heavy / copy
    );
    assert!(
        heavy / copy > 2.0,
        "control did not register as compute-bound: the harness cannot see compute"
    );
    let mut out = Vec::new();
    for (name, op) in UNARY_OPS {
        let def = OpDef::elementwise(
            &format!("def_{name}"),
            1,
            &[ElementKind::F32],
            input(0).unary(*op),
        );
        let k = Kernel::from_unary_f32_cuda(&def);
        let ns = time(&k.source, &k.name)?;
        out.push(Timing {
            op: name.to_string(),
            ns_per_elem: ns,
            vs_copy: ns / copy,
        });
    }
    println!("copy {copy:.6} ns/elem (N={N})");
    Ok(out)
}
