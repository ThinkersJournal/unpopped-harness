// SPDX-License-Identifier: MIT OR Apache-2.0
use unpopped_harness::bench::{Timing, UNARY_OPS};
use unpopped_harness::gpu_bench::time_unary_ops_gpu;

#[test]
fn gpu_unary_timings_cover_every_op() {
    let t: Vec<Timing> = time_unary_ops_gpu(7, 50).unwrap();
    assert_eq!(t.len(), UNARY_OPS.len());
    for x in &t {
        println!(
            "{:8} {:8.5} ns/elem  {:7.2}x copy",
            x.op, x.ns_per_elem, x.vs_copy
        );
    }
    assert!(t
        .iter()
        .all(|x| x.ns_per_elem.is_finite() && x.ns_per_elem > 0.0));
}
