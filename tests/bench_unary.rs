// SPDX-License-Identifier: MIT OR Apache-2.0
use unpopped_harness::bench::{time_unary_ops, UNARY_OPS};

#[test]
fn unary_op_timings_cover_every_op() {
    let w = std::env::temp_dir().join("unpopped-harness-bench");
    let t = time_unary_ops(&w, 7, 200).unwrap();
    assert_eq!(t.len(), UNARY_OPS.len());
    let mut s = t.clone();
    s.sort_by(|a, b| a.vs_copy.partial_cmp(&b.vs_copy).unwrap());
    for x in &s {
        println!(
            "{:8} {:8.4} ns/elem  {:7.2}x copy",
            x.op, x.ns_per_elem, x.vs_copy
        );
    }
    assert!(t
        .iter()
        .all(|x| x.ns_per_elem.is_finite() && x.ns_per_elem > 0.0));
}
