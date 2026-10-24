// SPDX-License-Identifier: MIT OR Apache-2.0
//! Print the stage profile and the emitter hotspot map: `cargo run --release --example profile`.

use unpopped_harness::profile::{hotspots, profile_unary, HOTSPOT_RATIO};

fn main() {
    let p = profile_unary(200);
    for r in &p {
        print!(
            "{:<8} ir {} roundtrip {:>7} ns",
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
    println!("\nhotspots (>= {HOTSPOT_RATIO}x the emitter's own median):");
    for h in hotspots(&p, HOTSPOT_RATIO) {
        println!("  {:<8} {:<6} {:.1}x", h.op, h.emitter, h.ratio);
    }
}
