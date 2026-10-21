// SPDX-License-Identifier: MIT OR Apache-2.0
use unpopped_harness::chain_bench::{chain_names, time_chains_cpu, time_chains_gpu};

#[test]
fn chains_cpu_vs_cuda() {
    let w = std::env::temp_dir().join("unpopped-harness-chain");
    let cpu = time_chains_cpu(&w, 7, 200).unwrap();
    let (dev, gpu) = time_chains_gpu(7, 50).unwrap();
    println!("device: {dev}");
    assert_eq!(cpu.len(), chain_names().len());
    assert_eq!(gpu.len(), cpu.len());
    println!(
        "{:14} {:>10} {:>9} | {:>10} {:>9} | cpu/gpu",
        "chain", "cpu ns", "x copy", "gpu ns", "x copy"
    );
    for (c, g) in cpu.iter().zip(&gpu) {
        assert_eq!(c.name, g.name);
        println!(
            "{:14} {:10.4} {:8.2}x | {:10.5} {:8.2}x | {:8.1}",
            c.name,
            c.ns_per_elem,
            c.vs_copy,
            g.ns_per_elem,
            g.vs_copy,
            c.ns_per_elem / g.ns_per_elem
        );
    }
}
