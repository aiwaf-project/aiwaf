//! Stable "fuzz smoke" tests for the request parsers.
//!
//! These throw random and mutated inputs at the SQL tokenizer, the request walker and the route
//! checks, and fail on a panic or a slow input. They run with every `cargo test`, so they're
//! short. The real fuzz targets live in `rust/fuzz` and run in CI on nightly.

use crate::payload::inspect_sql;
use crate::route_risks::{RouteRisks, SINKS, rule_for};
use crate::sql::{classify, classify_fields};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

/// Pieces that real attacks and parsers care about, so mutations reach deep code paths.
const PIECES: &[&str] = &[
    "'", "\"", "`", "\\", "--", "#", "/*", "*/", "/*!50000", ";", "(", ")", ",", "=", "<>", "||", "&&",
    "+", "%", "%27", "%2527", "%u2019", "%00", "\0", "\u{2019}", "\u{a0}", "\u{0b}", " ", "\n", "OR", "or", "AND",
    "UNION", "SELECT", "ALL", "FROM", "WHERE", "1", "-1", "1.5e3", "0x1F", "NULL", "sleep(", "pg_sleep(5)",
    "information_schema", "{", "}", "[", "]", ":", "\"a\":", "<", ">", "<a b=\"", "\">", "&", "=1&", "$ne", "../",
    "{{", "}}", "${", "`id`", "|", "é", "😀", "\u{d7ff}", "\u{fffd}", "NaN", "Infinity",
];

fn sample(rng: &mut StdRng, seeds: &[String]) -> String {
    let mut text = if rng.gen_bool(0.5) && !seeds.is_empty() {
        seeds[rng.gen_range(0..seeds.len())].clone()
    } else {
        String::new()
    };
    for _ in 0..rng.gen_range(0..24) {
        let piece = PIECES[rng.gen_range(0..PIECES.len())];
        let position = text.char_indices().map(|(i, _)| i).chain(std::iter::once(text.len())).nth(
            rng.gen_range(0..=text.chars().count()),
        );
        text.insert_str(position.unwrap_or(text.len()), piece);
    }
    if rng.gen_bool(0.05) {
        // Occasionally make it long, to catch anything slower than linear.
        let piece = PIECES[rng.gen_range(0..PIECES.len())];
        text.push_str(&piece.repeat(rng.gen_range(500..3000)));
    }
    text
}

fn seeds() -> Vec<String> {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("../../../../tests/fixtures/sql_injection_cases.json")).unwrap();
    let mut seeds = Vec::new();
    for key in ["attacks", "ordinary"] {
        for value in cases[key].as_array().unwrap() {
            seeds.push(value.as_str().unwrap().to_owned());
        }
    }
    seeds
}

fn timed<T>(limit: Duration, label: &str, input: &str, work: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = work();
    let elapsed = started.elapsed();
    assert!(elapsed < limit, "{label} took {elapsed:?} on {} chars: {:?}", input.chars().count(), input.chars().take(120).collect::<String>());
    result
}

// Generous so slow CI machines and debug builds pass; a quadratic path still blows well past it.
const LIMIT: Duration = Duration::from_millis(500);

#[test]
fn the_sql_tokenizer_never_panics_or_stalls() {
    let seeds = seeds();
    let mut rng = StdRng::seed_from_u64(0xA1AF);
    for _ in 0..4000 {
        let input = sample(&mut rng, &seeds);
        timed(LIMIT, "classify", &input, || classify(&input));
        timed(LIMIT, "classify_fields", &input, || classify_fields(&input));
    }
}

#[test]
fn the_request_walker_never_panics_or_stalls() {
    let seeds = seeds();
    let mut rng = StdRng::seed_from_u64(0xB0D1);
    for _ in 0..2000 {
        let query = sample(&mut rng, &seeds);
        let body = if rng.gen_bool(0.5) {
            serde_json::json!({"a": sample(&mut rng, &seeds), "b": [sample(&mut rng, &seeds)]}).to_string()
        } else {
            sample(&mut rng, &seeds)
        };
        let mut raw = body.clone().into_bytes();
        if rng.gen_bool(0.1) {
            // Arbitrary bytes, including invalid UTF-8 and UTF-16 byte order marks.
            let start = [&[0xff, 0xfe][..], &[0xfe, 0xff], &[0xef, 0xbb, 0xbf], &[0x80, 0xc0]][rng.gen_range(0..4)];
            raw.splice(0..0, start.iter().copied());
        }
        let budget = [16, 1024, 65536][rng.gen_range(0..3)];
        timed(LIMIT, "inspect_sql", &body, || inspect_sql(query.as_bytes(), &raw, budget, false));
    }
}

#[test]
fn route_checks_never_panic_or_stall() {
    let seeds = seeds();
    let mut rng = StdRng::seed_from_u64(0xC4EC);
    for _ in 0..300 {
        let mut routes = serde_json::Map::new();
        for _ in 0..rng.gen_range(1..6) {
            let prefix = sample(&mut rng, &[]).replace('/', "");
            let parameter = ["<p>", "{p}", ":p", "*", "**", "<path:p>"][rng.gen_range(0..6)];
            let route = format!("/{prefix}/{parameter}");
            let sink = SINKS[rng.gen_range(0..SINKS.len())];
            let input = ["", "p", "q"][rng.gen_range(0..3)];
            routes.insert(route, serde_json::json!({"injection_risks": [{"sink": sink, "input": input}]}));
        }
        let risks = RouteRisks::new(&serde_json::json!({"routes": routes}));
        for _ in 0..10 {
            let path = format!("/{}", sample(&mut rng, &seeds));
            let query = sample(&mut rng, &seeds);
            let body = sample(&mut rng, &seeds);
            timed(LIMIT, "route inspect", &body, || risks.inspect(&path, &query, body.as_bytes()));
            let sink = SINKS[rng.gen_range(0..SINKS.len())];
            timed(LIMIT, "rule_for", &query, || rule_for(sink, &query));
        }
    }
}

#[test]
fn worst_case_inputs_stay_fast() {
    for input in [
        "'".repeat(20_000),
        "\"".repeat(20_000),
        "--".repeat(10_000),
        "#".repeat(20_000),
        "/*".repeat(10_000),
        "(".repeat(20_000),
        "union ".repeat(5_000),
        "1 union select ".repeat(2_000),
        "a=".repeat(10_000),
        "&".repeat(20_000),
        "%u".repeat(10_000),
        "%25".repeat(10_000),
        "\\'".repeat(10_000),
        format!("{}'", "x".repeat(60_000)),
    ] {
        timed(Duration::from_secs(2), "classify_fields", &input, || classify_fields(&input));
        timed(Duration::from_secs(2), "inspect_sql", &input, || inspect_sql(input.as_bytes(), input.as_bytes(), 1 << 20, false));
    }
}
