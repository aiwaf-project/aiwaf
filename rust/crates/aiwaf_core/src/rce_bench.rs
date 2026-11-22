//! Timing for the remote code execution check: run with
//! cargo test --release -p aiwaf_core rce_bench -- --ignored --nocapture
#![cfg(test)]

use std::time::Instant;

fn per_call(label: &str, rounds: usize, work: impl Fn()) {
    work();
    let started = Instant::now();
    for _ in 0..rounds {
        work();
    }
    println!("{label}: {:?} each", started.elapsed() / rounds as u32);
}

#[test]
#[ignore]
fn timings() {
    let shared: serde_json::Value =
        serde_json::from_str(include_str!("../../../../tests/fixtures/ordinary_text.json")).unwrap();
    let ordinary: Vec<String> = shared["values"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect();
    let body = br#"{"name": "Ada Lovelace", "email": "ada@example.com", "age": 36, "city": "London",
        "bio": "Mathematician and writer, known for her work on the Analytical Engine.",
        "tags": ["math", "engines", "poetry"], "address": {"street": "12 St James's Square", "zip": "SW1Y 4JH"},
        "website": "https://example.com/ada", "phone": "+44 20 7946 0000", "notes": "Prefers email; replies within 2 days."}"#;
    let query = b"q=red+shoes&size=9&sort=price&page=2&utm_source=newsletter&ref=home";
    per_call("rce::classify over the ordinary corpus (per value)", 200, || {
        for value in &ordinary {
            std::hint::black_box(crate::rce::classify(value));
        }
    });
    println!("  ({} values)", ordinary.len());
    per_call("sql::classify over the ordinary corpus (per value)", 200, || {
        for value in &ordinary {
            std::hint::black_box(crate::sql::classify(value));
        }
    });
    per_call("inspect_sql: typical JSON body", 20000, || {
        std::hint::black_box(crate::payload::inspect_sql(b"", body, 65536, false));
    });
    per_call("inspect_sql: typical query string", 20000, || {
        std::hint::black_box(crate::payload::inspect_sql(query, b"", 65536, false));
    });
    crate::rce::BENCH_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
    per_call("inspect_sql without RCE: typical JSON body", 20000, || {
        std::hint::black_box(crate::payload::inspect_sql(b"", body, 65536, false));
    });
    per_call("inspect_sql without RCE: typical query string", 20000, || {
        std::hint::black_box(crate::payload::inspect_sql(query, b"", 65536, false));
    });
    crate::rce::BENCH_OFF.store(false, std::sync::atomic::Ordering::Relaxed);
    let form = b"name=Ada+Lovelace&email=ada%40example.com&city=London&bio=Mathematician+and+writer&tags=math&tags=poetry";
    per_call("inspect_sql: typical form body", 20000, || {
        std::hint::black_box(crate::payload::inspect_sql(b"", form, 65536, false));
    });
    let mut upload = b"--b\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\nHoliday photos\r\n--b\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"beach.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".to_vec();
    upload.extend(std::iter::repeat_n(7u8, 60_000));
    upload.extend_from_slice(b"\r\n--b--\r\n");
    per_call("  of which multipart_parts", 2000, || {
        std::hint::black_box(crate::payload::multipart_parts(&upload));
    });
    per_call("  of which classify_upload", 2000, || {
        std::hint::black_box(crate::rce::classify_upload(&upload[200..]));
    });
    per_call("inspect_sql: 60 KB multipart upload", 2000, || {
        std::hint::black_box(crate::payload::inspect_sql(b"", &upload, 65536, false));
    });
    let gzipped = {
        use std::io::Write;
        let mut out = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        out.write_all(body).unwrap();
        out.finish().unwrap()
    };
    per_call("inspect_sql: typical JSON body, gzip", 20000, || {
        std::hint::black_box(crate::payload::inspect_sql(b"", &gzipped, 65536, false));
    });
    let headers: Vec<(String, String)> = (0..20).map(|i| (format!("x-header-{i}"), "ordinary value".to_owned())).collect();
    per_call("inspect_sql_headers: 20 headers", 20000, || {
        std::hint::black_box(crate::payload::inspect_sql_headers(headers.iter().map(|(n, v)| (n.as_str(), v.as_str())), 65536));
    });
    let schema = crate::route_risks::RouteRisks::new(&serde_json::json!({"routes": {"/profile": {
        "inputs": {"name": "text", "email": "text", "age": "int", "city": "text", "bio": "text", "tags": "text",
                   "address": "text", "website": "text", "phone": "text", "notes": "text"}, "inputs_complete": true}}}));
    per_call("route schema review: clean request (type checks only)", 20000, || {
        std::hint::black_box(schema.review("/profile", "", body, 65536, None));
    });
    let flagged = br#"{"name": "Ada", "age": 36, "bio": "use sleep(5) in the query"}"#;
    per_call("route schema review: flagged request (re-check)", 20000, || {
        std::hint::black_box(schema.review("/profile", "", flagged, 65536, Some("sql_function")));
    });
    per_call("rce::classify: short word", 200000, || {
        std::hint::black_box(crate::rce::classify("shoes"));
    });
    per_call("rce::classify: email", 200000, || {
        std::hint::black_box(crate::rce::classify("ada@example.com"));
    });
}
