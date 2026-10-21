// SPDX-License-Identifier: MIT OR Apache-2.0
use baracuda_nvrtc::Program;
use unpopped::ir::{input, OpDef, UnaryOp};
use unpopped_harness::Kernel;
use unpopped_vocab::ElementKind;

#[test]
fn sqrt_add_ptx() {
    for d in [1usize, 2, 3, 4, 32] {
        let ops = [UnaryOp::Sqrt, UnaryOp::Sqr, UnaryOp::Sqrt];
        let mut e = input(0);
        for i in 0..d {
            e = e.unary(ops[i % 3]);
        }
        let def = OpDef::elementwise("chain", 1, &[ElementKind::F32], e);
        let k = Kernel::from_unary_f32_cuda(&def);
        let ptx = Program::compile(&k.source, "c.cu", &["--gpu-architecture=compute_89"]).unwrap();

        let n = |s: &str| {
            ptx.lines()
                .filter(|l| l.trim_start().starts_with(s))
                .count()
        };
        println!(
            "d={d}: src_len={} sqrt={} mul={} st.global={} total_lines={}",
            k.source.len(),
            n("sqrt"),
            n("mul"),
            n("st.global"),
            ptx.lines().count()
        );
        if d == 3 || d == 32 {
            println!("---- source d={d}\n{}\n---- ptx d={d}\n{ptx}", k.source);
        }
    }
}
