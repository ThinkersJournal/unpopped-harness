// SPDX-License-Identifier: MIT OR Apache-2.0
//! Emitter exercise on the shapes #74 (0.20.6) newly admitted: global Window pool,
//! Im2Col with N=1 / C=1, and top-1 with indices. Published crates only.
//! GPU (CUDA emitter, NVRTC) vs the Unpopped oracle, bit-exact, NaN sentinel.
use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};
use baracuda_nvrtc::Program;
use unpopped::generate;
use unpopped::ir::{OpDef, ReduceOp, SortOrder};
use unpopped::oracle::{evaluate, TypedBuffer};
use unpopped::plan::build_plan;
use unpopped_vocab::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

const F: ElementKind = ElementKind::F32;

fn d(s: &[i64], st: &[i64], k: ElementKind) -> OperandDesc {
    OperandDesc::new(s.len(), s, st, k, 4)
}

fn dense(s: &[i64]) -> Vec<i64> {
    let mut st = vec![1; s.len()];
    for i in (0..s.len().saturating_sub(1)).rev() {
        st[i] = st[i + 1] * s[i + 1];
    }
    st
}

/// Launch `src`/`name` with i64 `args` after (in0, out); return `out_len` f32 cells
/// pre-filled with a NaN sentinel.
fn run_gpu(
    src: &str,
    name: &str,
    input: &[f32],
    out_len: usize,
    args: &[i64],
    threads: usize,
) -> Vec<f32> {
    baracuda_driver::init().unwrap();
    let dev = Device::get(0).unwrap();
    let (maj, min) = dev.compute_capability().unwrap();
    let arch = format!("--gpu-architecture=compute_{maj}{min}");
    let ptx = Program::compile(src, "k.cu", &[&arch]).expect("nvrtc");
    let ctx = Context::new(&dev).unwrap();
    let stream = Stream::new(&ctx).unwrap();
    let module = Module::load_ptx(&ctx, &ptx).unwrap();
    let k = module.get_function(name).unwrap();
    let d_in = DeviceBuffer::<f32>::new(&ctx, input.len()).unwrap();
    let d_out = DeviceBuffer::<f32>::new(&ctx, out_len).unwrap();
    d_in.copy_from_host(input).unwrap();
    d_out
        .copy_from_host(&vec![f32::from_bits(0x7fc0_dead); out_len])
        .unwrap();
    let (ip, op_) = (d_in.as_raw(), d_out.as_raw());
    let block = 128u32;
    let grid = (threads as u32).div_ceil(block).max(1);
    unsafe {
        let mut l = k
            .launch()
            .grid((grid, 1, 1))
            .block((block, 1, 1))
            .stream(&stream)
            .arg(&ip)
            .arg(&op_);
        for a in args {
            l = l.arg(a);
        }
        l.launch().unwrap();
    }
    stream.synchronize().unwrap();
    let mut out = vec![0f32; out_len];
    d_out.copy_to_host(&mut out).unwrap();
    out
}

fn check(tag: &str, got: &[f32], want: &TypedBuffer) {
    let w = want.to_f64_vec();
    assert_eq!(got.len(), w.len(), "{tag}: length");
    let mut bad = vec![];
    for i in 0..got.len() {
        let wb = (w[i] as f32).to_bits();
        if got[i].to_bits() != wb
            && !(got[i].is_nan() && (w[i] as f32).is_nan() && got[i].to_bits() != 0x7fc0_dead)
        {
            bad.push((i, got[i], w[i]));
        }
    }
    assert!(
        bad.is_empty(),
        "{tag}: {} of {} cells differ, first {:?}",
        bad.len(),
        got.len(),
        &bad[..bad.len().min(6)]
    );
}

fn data(n: usize) -> Vec<f32> {
    // Distinct, mixed-sign, with a -0.0 and a NaN so ties/NaN paths are touched.
    (0..n)
        .map(|i| ((i * 7919 % 101) as f32 - 50.0) * 0.25)
        .collect()
}

fn window_case(
    tag: &str,
    (rows, len, out_len): (i64, i64, i64),
    op: ReduceOp,
    size: u8,
    stride: u8,
    pad: u8,
) {
    let o = OpDef::window_simple("w", &[F], op, 1, size, stride, 1, pad, pad, false);
    let ops = [
        d(&[rows, len], &dense(&[rows, len]), F),
        d(&[rows, out_len], &dense(&[rows, out_len]), F),
    ];
    let key = structure_key(OpCategory::Pooling, &ops, ArchSku::Sm89);
    let plan = build_plan(&o, &key);
    let mut x = data((rows * len) as usize);
    x[1] = f32::from_bits(0x8000_0000);
    let want = evaluate(&plan, &ops, &[TypedBuffer::from_f32(&[rows, len], &x)], &[]);
    let k = generate(&o, &key, &baracuda_cuda_emit::Cuda);
    let got = run_gpu(
        &k.source,
        &k.name,
        &x,
        (rows * out_len) as usize,
        &[rows, len, out_len],
        (rows * out_len) as usize,
    );
    check(tag, &got, &want[0]);
}

#[test]
fn global_window_pool_matches_the_oracle() {
    // The newly admitted shape: one window spans the whole row, out extent 1.
    for op in [ReduceOp::Max, ReduceOp::Min, ReduceOp::Sum] {
        window_case(&format!("gpool-{op:?}-3x8"), (3, 8, 1), op, 8, 8, 0);
        window_case(&format!("gpool-{op:?}-1x8"), (1, 8, 1), op, 8, 8, 0);
        window_case(&format!("gpool-{op:?}-5x13"), (5, 13, 1), op, 13, 13, 0);
    }
}

#[test]
fn window_control_non_global_pool_matches() {
    // Control: the pre-0.20.6 shape (out extent > 1) through the same runner.
    window_case("ctl-max-3x8", (3, 8, 4), ReduceOp::Max, 2, 2, 0);
}

fn im2col_case(tag: &str, shape: [i64; 4], kern: (u8, u8), stride: (u8, u8), pad: (u8, u8)) {
    let [n, c, h, w] = shape;
    let o = OpDef::im2col_2d("i", F, kern, stride, pad, (1, 1));
    let oh = (h + 2 * pad.0 as i64 - (kern.0 as i64 - 1) - 1) / stride.0 as i64 + 1;
    let ow = (w + 2 * pad.1 as i64 - (kern.1 as i64 - 1) - 1) / stride.1 as i64 + 1;
    let rows = c * kern.0 as i64 * kern.1 as i64;
    let os = [n, rows, oh * ow];
    let ops = [d(&shape, &dense(&shape), F), d(&os, &dense(&os), F)];
    let key = structure_key(OpCategory::Convolution, &ops, ArchSku::Sm89);
    let plan = build_plan(&o, &key);
    let x = data((n * c * h * w) as usize);
    let want = evaluate(&plan, &ops, &[TypedBuffer::from_f32(&shape, &x)], &[]);
    let k = generate(&o, &key, &baracuda_cuda_emit::Cuda);
    let total = (n * rows * oh * ow) as usize;
    let got = run_gpu(&k.source, &k.name, &x, total, &[n, c, h, w, oh, ow], total);
    check(tag, &got, &want[0]);
}

#[test]
fn im2col_size_one_batch_or_channel_matches_the_oracle() {
    im2col_case("n1c2", [1, 2, 5, 4], (2, 2), (1, 1), (1, 0));
    im2col_case("n1c1", [1, 1, 4, 4], (3, 3), (1, 1), (1, 1));
    im2col_case("n2c1", [2, 1, 5, 4], (3, 2), (2, 1), (0, 1));
    // Control: the always-admitted shape.
    im2col_case("ctl-n2c3", [2, 3, 5, 4], (2, 3), (1, 1), (0, 0));
}

#[test]
fn top1_with_indices_is_refused_by_the_published_cuda_emitter() {
    let o = OpDef::row_sort_indices("top1", F, SortOrder::Desc);
    let ops = [
        d(&[3, 8], &[8, 1], F),
        d(&[3, 1], &[1, 1], F),
        d(&[3, 1], &[1, 1], ElementKind::I32),
    ];
    let key = structure_key(OpCategory::Sorting, &ops, ArchSku::Sm89);
    let _ = build_plan(&o, &key); // unpopped admits it
    let r = std::panic::catch_unwind(|| generate(&o, &key, &baracuda_cuda_emit::Cuda).source);
    let msg = r
        .err()
        .map(|e| e.downcast_ref::<String>().cloned().unwrap_or_default());
    println!("TOP1 emitter result: {msg:?}");
    // Control: a top-2 of the same op generates.
    let ops2 = [
        d(&[3, 8], &[8, 1], F),
        d(&[3, 2], &[2, 1], F),
        d(&[3, 2], &[2, 1], ElementKind::I32),
    ];
    let key2 = structure_key(OpCategory::Sorting, &ops2, ArchSku::Sm89);
    let _ = build_plan(&o, &key2);
    let ok = generate(&o, &key2, &baracuda_cuda_emit::Cuda).source;
    assert!(!ok.is_empty());
    assert!(msg.is_some(), "emitter now accepts top-1");
}
