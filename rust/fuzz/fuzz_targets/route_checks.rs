//! Route checks must never panic on any manifest, path, query or body.
#![no_main]
use libfuzzer_sys::fuzz_target;

const SINKS: &[&str] = aiwaf_core::route_risks::SINKS;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let mut parts = text.splitn(4, '\u{1}');
    let (route, path, query, body) = (
        parts.next().unwrap_or_default(),
        parts.next().unwrap_or_default(),
        parts.next().unwrap_or_default(),
        parts.next().unwrap_or_default(),
    );
    let sink = SINKS[data.first().map_or(0, |b| usize::from(*b) % SINKS.len())];
    let manifest = serde_json::json!({"routes": {route: {"injection_risks": [{"sink": sink, "input": ""}]}}});
    let risks = aiwaf_core::route_risks::RouteRisks::new(&manifest);
    let _ = risks.inspect(path, query, body.as_bytes());
    let _ = aiwaf_core::route_risks::rule_for(sink, body);
    // Arbitrary manifest JSON too.
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) {
        let _ = aiwaf_core::route_risks::RouteRisks::new(&value).inspect(path, query, body.as_bytes());
    }
});
