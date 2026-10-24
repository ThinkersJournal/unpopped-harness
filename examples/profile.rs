// SPDX-License-Identifier: MIT OR Apache-2.0
//! Print the stage profile and the emitter hotspot map: `cargo run --release --example profile`.

use unpopped_harness::profile::{
    hotspots, profile_binary, profile_ternary, profile_unary, OpProfile, HOTSPOT_RATIO,
};

fn report(kind: &str, p: &[OpProfile]) {
    println!("== {kind} ({} ops)", p.len());
    for r in p {
        print!(
            "{:<10} ir {} roundtrip {:>7} ns",
            r.op,
            &r.ir_hash[..15],
            r.text_roundtrip_ns
        );
        for e in &r.emitters {
            match &e.declined {
                None => print!("  {} {:>7} ns", e.emitter, e.emit_ns),
                Some(_) => print!("  {} DECLINED", e.emitter),
            }
        }
        println!();
    }
    println!("hotspots (>= {HOTSPOT_RATIO}x the emitter's own median):");
    for h in hotspots(p, HOTSPOT_RATIO) {
        println!("  {:<10} {:<6} {:.1}x", h.op, h.emitter, h.ratio);
    }
}

fn main() {
    report("unary", &profile_unary(200));
    report("binary", &profile_binary(200));
    report("ternary", &profile_ternary(200));
}
