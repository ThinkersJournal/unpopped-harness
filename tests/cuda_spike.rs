// SPDX-License-Identifier: MIT OR Apache-2.0
//! Second-target spike: unpopped Cuda backend -> NVRTC -> driver launch on the real GPU,
//! compared bit-for-bit with relu computed on the host. One kernel, a spread of inputs
//! including NaN, -0.0 and the infinities. Proves the path runs; it is not the harness.

use baracuda_cuda_emit::Cuda;
use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};
use baracuda_nvrtc::Program;
use unpopped::generate;
use unpopped::ir::{input, OpDef, UnaryOp};
use unpopped_vocab::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

fn relu_host(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

fn inputs() -> Vec<f32> {
    let mut v: Vec<f32> = (0u64..(1 << 32))
        .step_by(4099)
        .map(|b| f32::from_bits(b as u32))
        .collect();
    for b in [
        0x8000_0000u32,
        0x0000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0000,
        0x7f80_dead,
    ] {
        v.push(f32::from_bits(b));
    }
    v
}

#[test]
fn cuda_relu_matches_host_bit_for_bit() {
    let op = OpDef::elementwise(
        "relu_k",
        1,
        &[ElementKind::F32],
        input(0).unary(UnaryOp::Relu),
    );
    let d = OperandDesc::new(1, &[65535], &[1], ElementKind::F32, 4);
    let key = structure_key(OpCategory::UnaryElementwise, &[d, d], ArchSku::Sm89);
    let k = generate(&op, &key, &Cuda);

    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let (maj, min) = device.compute_capability().unwrap();
    println!("device: {} cc {maj}.{min}", device.name().unwrap());
    let arch = format!("--gpu-architecture=compute_{maj}{min}");
    let ptx = Program::compile(&k.source, "relu_k.cu", &[&arch]).unwrap();

    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();
    let module = Module::load_ptx(&ctx, &ptx).unwrap();
    let kernel = module.get_function(&k.name).unwrap();

    let xs = inputs();
    let n = xs.len() as i64;
    let d_in = DeviceBuffer::from_slice(&ctx, &xs).unwrap();
    // Sentinel-fill the output so "never wrote" cannot pass as a match.
    let d_out =
        DeviceBuffer::from_slice(&ctx, &vec![f32::from_bits(0x7fc0_dead); xs.len()]).unwrap();
    let (ip, op_) = (d_in.as_raw(), d_out.as_raw());
    let block = 256u32;
    let grid = (xs.len() as u32).div_ceil(block);
    unsafe {
        kernel
            .launch()
            .grid((grid, 1, 1))
            .block((block, 1, 1))
            .stream(&stream)
            .arg(&ip)
            .arg(&op_)
            .arg(&n)
            .launch()
            .unwrap();
    }
    stream.synchronize().unwrap();
    let mut got = vec![0f32; xs.len()];
    d_out.copy_to_host(&mut got).unwrap();

    let mut bad = 0usize;
    for (x, g) in xs.iter().zip(&got) {
        let want = relu_host(*x);
        let same = (want.is_nan() && g.is_nan()) || want.to_bits() == g.to_bits();
        if !same {
            bad += 1;
            if bad <= 5 {
                println!(
                    "MISMATCH in={:#010x} want={:#010x} got={:#010x}",
                    x.to_bits(),
                    want.to_bits(),
                    g.to_bits()
                );
            }
        }
    }
    println!("checked {} inputs, {bad} mismatches", xs.len());
    assert_eq!(bad, 0);
}
