//! The request walker must never panic on any query string and body bytes.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The first byte picks the split and the size budget, so one input covers both parts.
    let Some((&control, rest)) = data.split_first() else { return };
    let split = usize::from(control) * rest.len() / 255;
    let (query, body) = rest.split_at(split.min(rest.len()));
    let budget = [16, 512, 65536][usize::from(control) % 3];
    let _ = aiwaf_core::payload::inspect_sql(query, body, budget, false);
});
