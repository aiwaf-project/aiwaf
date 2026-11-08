//! Pure enforcement decisions. Framework request/session extraction stays in adapters.
use pyo3::prelude::*;
use regex::Regex;
use std::{collections::HashSet, sync::OnceLock};

#[pyfunction]
pub fn policy_rate(
    timestamps: Vec<f64>,
    now: f64,
    window: f64,
    max_requests: i64,
    flood: i64,
) -> (String, usize, Vec<f64>) {
    let mut trimmed: Vec<f64> = timestamps
        .into_iter()
        .filter(|t| now - t < window.max(1.0))
        .collect();
    trimmed.push(now);
    let count = trimmed.len();
    let action = if count as i64 > flood {
        "flood_block"
    } else if count as i64 > max_requests {
        "throttle"
    } else {
        "allow"
    };
    (action.to_owned(), count, trimmed)
}

#[pyfunction]
pub fn policy_geo(
    country: String,
    allow: HashSet<String>,
    block: HashSet<String>,
    dynamic: HashSet<String>,
) -> (bool, String, String) {
    let country = country.trim().to_uppercase();
    let blocked = !country.is_empty()
        && if !allow.is_empty() {
            !allow.contains(&country)
        } else {
            block.contains(&country) || dynamic.contains(&country)
        };
    let reason = if blocked {
        format!("Geo blocked: {country}")
    } else {
        String::new()
    };
    (blocked, country, reason)
}

#[pyfunction]
#[pyo3(signature = (now, get_time, path, min_time, max_time))]
pub fn policy_form(
    now: f64,
    get_time: Option<f64>,
    path: String,
    min_time: f64,
    max_time: f64,
) -> (String, Option<String>, Option<i32>, Option<String>) {
    let Some(get_time) = get_time else {
        return ("allow".into(), None, None, None);
    };
    let elapsed = now - get_time;
    if elapsed > max_time {
        return (
            "page_expired".into(),
            None,
            Some(409),
            Some("Page has expired. Please reload and try again.".into()),
        );
    }
    let path = path.to_lowercase();
    let minimum = if [
        "/admin/login/",
        "/login/",
        "/accounts/login/",
        "/auth/login/",
        "/signin/",
    ]
    .iter()
    .any(|s| path.starts_with(s))
    {
        0.1
    } else {
        min_time
    };
    if elapsed < minimum {
        return (
            "block".into(),
            Some(format!("Form submitted too quickly ({elapsed:.2}s)")),
            Some(403),
            None,
        );
    }
    ("allow".into(), None, None, None)
}

#[pyfunction]
pub fn policy_uuid(text: String) -> bool {
    // Match Python's length/pattern check followed by uuid.UUID: misplaced hyphens
    // are accepted only when removing them leaves exactly 32 hexadecimal digits.
    let text = text.trim().to_lowercase();
    text.len() == 36
        && text.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
        && text.chars().filter(|c| *c != '-').count() == 32
}

#[pyfunction]
pub fn policy_reputation(
    score: i64,
    offenses: i64,
    old_reasons: Vec<String>,
    reason: String,
) -> (i64, i64, Vec<String>, bool, Option<i64>) {
    let normalized = if reason.trim().is_empty() {
        "unknown"
    } else {
        reason.trim()
    };
    let mut reasons = Vec::new();
    let mut seen = HashSet::new();
    for raw in old_reasons
        .into_iter()
        .chain(std::iter::once(normalized.to_owned()))
    {
        let value = if raw.trim().is_empty() {
            "unknown".to_owned()
        } else {
            raw.trim().to_owned()
        };
        if seen.insert(value.to_lowercase()) {
            reasons.push(value);
        }
    }
    let lower = normalized.to_lowercase();
    let weight = [
        ("scanner", 20),
        ("scan", 20),
        ("sqli", 40),
        ("sql injection", 40),
        ("xss", 30),
        ("bruteforce", 25),
        ("brute force", 25),
        ("flood", 25),
        ("rate limit", 20),
        ("honeypot", 30),
        ("uuid", 25),
        ("header", 15),
        ("geo", 20),
        ("keyword", 20),
    ]
    .into_iter()
    .find(|(token, _)| lower.contains(token))
    .map(|(_, w)| w)
    .unwrap_or(10);
    let score = (score + weight).min(100);
    let offenses = offenses + 1;
    let duration = if score < 60 {
        None
    } else if score >= 80 || offenses >= 3 {
        Some(86400)
    } else if offenses == 2 {
        Some(3600)
    } else {
        Some(900)
    };
    (score, offenses, reasons, score >= 60, duration)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn policy_keyword(
    path: String,
    query_keys: HashSet<String>,
    exists: bool,
    learning: bool,
    static_kw: HashSet<String>,
    dynamic_kw: HashSet<String>,
    legitimate: HashSet<String>,
    exempt: HashSet<String>,
    prefixes: Vec<String>,
    malicious: HashSet<String>,
    contexts: HashSet<String>,
) -> (Option<String>, Vec<String>, Vec<String>) {
    static SPLIT: OnceLock<Regex> = OnceLock::new();
    static PROBE: OnceLock<Regex> = OnceLock::new();
    let path = path.to_lowercase();
    let normalized = path.trim_start_matches('/');
    let segments: Vec<String> = SPLIT
        .get_or_init(|| Regex::new(r"\W+").unwrap())
        .split(normalized)
        .filter(|s| s.chars().count() > 3)
        .map(str::to_owned)
        .collect();
    let learned: Vec<String> = if learning && !exists {
        segments
            .iter()
            .filter(|s| !legitimate.contains(*s) && !exempt.contains(*s) && contexts.contains(*s))
            .cloned()
            .collect()
    } else {
        Vec::new()
    };
    if !exists && PROBE.get_or_init(|| Regex::new(r"(^|/)\.(env|git|htaccess|htpasswd)(/|$)|\.(php|asp|aspx|jsp|cgi|bak|sql)(/|$)|xmlrpc\.php").unwrap()).is_match(&path) {
        return (Some("Keyword block: Inherently suspicious: probe path".into()), learned, segments);
    }
    let safe = prefixes
        .iter()
        .any(|p| !p.is_empty() && normalized.starts_with(p));
    let suspicious: HashSet<String> = static_kw
        .union(&dynamic_kw)
        .filter(|kw| {
            !exempt.contains(*kw)
                && !(legitimate.contains(*kw) && exists && !contexts.contains(*kw))
                && !safe
        })
        .cloned()
        .collect();
    let strong_count = [
        path.contains("../"),
        path.contains("..\\"),
        ["cmd", "exec", "system"]
            .iter()
            .any(|k| query_keys.contains(*k)),
        path.chars().filter(|c| *c == '%').count() > 5,
        segments.iter().filter(|s| malicious.contains(*s)).count() > 2,
    ]
    .into_iter()
    .filter(|v| *v)
    .count();
    let strong = strong_count >= 2
        || [
            "union+select",
            "drop+table",
            "<script",
            "javascript:",
            "onload=",
            "onerror=",
            "${",
            "{{",
            "eval(",
        ]
        .iter()
        .any(|s| path.contains(s));
    for segment in &segments {
        let reason = if suspicious.contains(segment) {
            Some(format!("Learned keyword: {segment}"))
        } else if !exists
            && !legitimate.contains(segment)
            && (contexts.contains(segment)
                || [
                    "hack",
                    "exploit",
                    "attack",
                    "malicious",
                    "evil",
                    "backdoor",
                    "inject",
                    "xss",
                ]
                .iter()
                .any(|s| segment.contains(s)))
        {
            Some(format!("Inherently suspicious: {segment}"))
        } else {
            None
        };
        if let Some(reason) = reason {
            if (!exists || strong) && (contexts.contains(segment) || !exists) {
                return (Some(format!("Keyword block: {reason}")), learned, segments);
            }
        }
    }
    (None, learned, segments)
}

/// Flask's context checks operate on plain request data, without Python callbacks.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn policy_keyword_flask(
    py: Python<'_>,
    path: String,
    query: String,
    query_keys: HashSet<String>,
    exists: bool,
    learning: bool,
    static_kw: HashSet<String>,
    dynamic_kw: HashSet<String>,
    legitimate: HashSet<String>,
    exempt: HashSet<String>,
    prefixes: Vec<String>,
    malicious: HashSet<String>,
) -> (Option<String>, Vec<String>, Vec<String>) {
    py.allow_threads(|| {
        let path = path.to_lowercase();
        let query = query.to_lowercase();
        static SPLIT: OnceLock<Regex> = OnceLock::new();
        let mut candidates: HashSet<String> = SPLIT
            .get_or_init(|| Regex::new(r"\W+").unwrap())
            .split(path.trim_start_matches('/'))
            .filter(|s| s.chars().count() > 3)
            .map(str::to_owned)
            .collect();
        candidates.extend(static_kw.iter().cloned());
        candidates.extend(dynamic_kw.iter().cloned());
        let traversal = path.contains("../") || path.contains("..\\");
        let query_attack = [
            "union", "select", "drop", "insert", "script", "alert", "eval",
        ]
        .iter()
        .any(|s| query.contains(s));
        let contexts = candidates
            .into_iter()
            .filter(|s| {
                (query_attack && query.contains(s.as_str()))
                    || traversal
                    || (!exists
                        && (s.starts_with('.')
                            || [".php", ".asp", ".jsp", ".cgi"]
                                .iter()
                                .any(|ext| s.contains(ext))))
            })
            .collect();
        policy_keyword(
            path, query_keys, exists, learning, static_kw, dynamic_kw, legitimate, exempt,
            prefixes, malicious, contexts,
        )
    })
}

#[pyfunction]
pub fn policy_sql_signature(py: Python<'_>, text: String) -> Option<String> {
    py.allow_threads(|| sql_signature(&text))
}

/// Returns the SQL tokenizer's rule for a value. The name is kept from the signature rules it
/// replaced, so callers and older Python releases keep working.
pub(crate) fn sql_signature(text: &str) -> Option<String> {
    aiwaf_core::sql::classify_fields(text).map(str::to_owned)
}

/// Checks a request's query and buffered body for SQL injection. Returns (rule, status):
/// 403 for an attack and 413 when the request is too large to inspect, or None.
#[pyfunction]
pub fn policy_sql_inspect(py: Python<'_>, query: Vec<u8>, body: Vec<u8>, max_bytes: usize) -> Option<(String, i32)> {
    py.allow_threads(|| aiwaf_core::payload::inspect_sql(&query, &body, max_bytes, false))
}

/// Checks request headers for SQL injection: cookie names and values, the Referer's query
/// string and path, and any other header except those holding credentials, framing or
/// browser-set values. Takes
/// (name, value) pairs and returns (rule, status) or None, like `policy_sql_inspect`.
#[pyfunction]
pub fn policy_sql_inspect_headers(py: Python<'_>, headers: Vec<(String, String)>, max_bytes: usize) -> Option<(String, i32)> {
    py.allow_threads(|| {
        aiwaf_core::payload::inspect_sql_headers(headers.iter().map(|(n, v)| (n.as_str(), v.as_str())), max_bytes)
    })
}

/// Checks a URL path for SQL injection, one segment at a time.
#[pyfunction]
pub fn policy_sql_inspect_path(py: Python<'_>, path: String, max_bytes: usize) -> Option<(String, i32)> {
    py.allow_threads(|| aiwaf_core::payload::inspect_sql_path(&path, max_bytes))
}

/// Returns the first route-check rule for this operation that matches the value.
#[pyfunction]
pub fn policy_route_rule(py: Python<'_>, sink: String, value: String) -> Option<String> {
    py.allow_threads(|| aiwaf_core::route_risks::rule_for(&sink, &value))
}

/// Route-specific checks compiled from a manifest's saved audit findings.
#[pyclass(name = "RouteRiskChecker")]
pub struct RouteRiskChecker {
    risks: aiwaf_core::route_risks::RouteRisks,
}

#[pymethods]
impl RouteRiskChecker {
    /// Compiles a manifest given as JSON text. Text that isn't a manifest gives no checks.
    #[new]
    fn new(manifest_json: &str) -> Self {
        let manifest = serde_json::from_str(manifest_json).unwrap_or(serde_json::Value::Null);
        Self { risks: aiwaf_core::route_risks::RouteRisks::new(&manifest) }
    }

    #[getter]
    fn empty(&self) -> bool {
        self.risks.is_empty()
    }

    /// Returns (route, sink, input, rule, confidence) for each match.
    fn inspect(
        &self,
        py: Python<'_>,
        path: &str,
        query: &str,
        body: Vec<u8>,
    ) -> Vec<(String, String, String, String, String)> {
        py.allow_threads(|| {
            self.risks
                .inspect(path, query, &body)
                .into_iter()
                .map(|e| (e.route, e.sink, e.input, e.rule, e.confidence))
                .collect()
        })
    }
}

/// Returns the tokenizer's rule when a value breaks out of a SQL value.
#[pyfunction]
pub fn policy_sql_tokenizer(py: Python<'_>, text: String) -> Option<String> {
    py.allow_threads(|| aiwaf_core::sql::classify(&text).map(str::to_owned))
}

/// Returns the first tokenizer rule across values, crossing the binding once.
#[pyfunction]
pub fn policy_sql_tokenizers(py: Python<'_>, texts: Vec<String>) -> Option<String> {
    py.allow_threads(|| texts.iter().find_map(|text| aiwaf_core::sql::classify(text)).map(str::to_owned))
}

/// Preserve first-value/first-rule order while crossing the binding once.
#[pyfunction]
pub fn policy_sql_signatures(py: Python<'_>, texts: Vec<String>) -> Option<String> {
    py.allow_threads(|| texts.iter().find_map(|text| sql_signature(text)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(policy_sql_signature, m)?)?;
    m.add_function(wrap_pyfunction!(policy_sql_signatures, m)?)?;
    m.add_function(wrap_pyfunction!(policy_sql_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(policy_sql_inspect, m)?)?;
    m.add_function(wrap_pyfunction!(policy_sql_inspect_headers, m)?)?;
    m.add_function(wrap_pyfunction!(policy_sql_inspect_path, m)?)?;
    m.add_function(wrap_pyfunction!(policy_route_rule, m)?)?;
    m.add_class::<RouteRiskChecker>()?;
    m.add_function(wrap_pyfunction!(policy_sql_tokenizers, m)?)?;
    m.add_function(wrap_pyfunction!(policy_keyword_flask, m)?)?;
    m.add_function(wrap_pyfunction!(policy_rate, m)?)?;
    m.add_function(wrap_pyfunction!(policy_geo, m)?)?;
    m.add_function(wrap_pyfunction!(policy_form, m)?)?;
    m.add_function(wrap_pyfunction!(policy_uuid, m)?)?;
    m.add_function(wrap_pyfunction!(policy_reputation, m)?)?;
    m.add_function(wrap_pyfunction!(policy_keyword, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(values: &[&str]) -> HashSet<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    fn keyword(
        path: &str,
        exists: bool,
        static_kw: &[&str],
        contexts: &[&str],
    ) -> (Option<String>, Vec<String>, Vec<String>) {
        policy_keyword(
            path.into(),
            set(&[]),
            exists,
            false,
            set(static_kw),
            set(&[]),
            set(&[]),
            set(&[]),
            vec![],
            set(&[]),
            set(contexts),
        )
    }

    #[test]
    fn rate_escalates_from_allow_to_throttle_to_flood() {
        assert_eq!(policy_rate(vec![99.0], 100.0, 60.0, 2, 3).0, "allow");
        assert_eq!(policy_rate(vec![98.0, 99.0], 100.0, 60.0, 2, 3).0, "throttle");
        assert_eq!(policy_rate(vec![97.0, 98.0, 99.0], 100.0, 60.0, 2, 3).0, "flood_block");
    }

    #[test]
    fn rate_drops_timestamps_outside_the_window() {
        let (action, count, kept) = policy_rate(vec![40.0, 41.0], 100.0, 60.0, 5, 10);
        assert_eq!((action.as_str(), count), ("allow", 2));
        assert_eq!(kept, vec![41.0, 100.0]);
    }

    #[test]
    fn rate_window_is_never_below_one_second() {
        assert_eq!(policy_rate(vec![99.5], 100.0, 0.0, 5, 10).1, 2);
        assert_eq!(policy_rate(vec![99.0], 100.0, 0.0, 5, 10).1, 1);
    }

    #[test]
    fn geo_allow_list_overrides_block_lists() {
        let (blocked, _, _) = policy_geo("FR".into(), set(&["US"]), set(&[]), set(&[]));
        assert!(blocked);
        let (blocked, _, reason) = policy_geo("US".into(), set(&["US"]), set(&["US"]), set(&[]));
        assert!(!blocked);
        assert_eq!(reason, "");
    }

    #[test]
    fn geo_normalizes_country_and_checks_dynamic_blocks() {
        let (blocked, country, reason) = policy_geo(" ru ".into(), set(&[]), set(&[]), set(&["RU"]));
        assert!(blocked);
        assert_eq!(country, "RU");
        assert_eq!(reason, "Geo blocked: RU");
    }

    #[test]
    fn geo_never_blocks_an_unknown_country() {
        assert!(!policy_geo("  ".into(), set(&["US"]), set(&[]), set(&[])).0);
    }

    #[test]
    fn form_without_a_page_view_is_allowed() {
        assert_eq!(policy_form(100.0, None, "/form".into(), 1.0, 300.0).0, "allow");
    }

    #[test]
    fn form_expires_after_max_time() {
        let (action, _, status, message) = policy_form(500.0, Some(100.0), "/form".into(), 1.0, 300.0);
        assert_eq!((action.as_str(), status), ("page_expired", Some(409)));
        assert!(message.unwrap().contains("expired"));
    }

    #[test]
    fn form_blocks_submissions_faster_than_min_time() {
        let (action, reason, status, _) = policy_form(100.5, Some(100.0), "/form".into(), 1.0, 300.0);
        assert_eq!((action.as_str(), status), ("block", Some(403)));
        assert_eq!(reason.unwrap(), "Form submitted too quickly (0.50s)");
        assert_eq!(policy_form(101.0, Some(100.0), "/form".into(), 1.0, 300.0).0, "allow");
    }

    #[test]
    fn login_forms_use_a_shorter_minimum() {
        for path in ["/login/", "/ACCOUNTS/LOGIN/", "/admin/login/", "/signin/"] {
            assert_eq!(policy_form(100.5, Some(100.0), path.into(), 1.0, 300.0).0, "allow", "{path}");
        }
        assert_eq!(policy_form(100.05, Some(100.0), "/login/".into(), 1.0, 300.0).0, "block");
    }

    #[test]
    fn uuid_requires_36_chars_and_32_hex_digits() {
        assert!(policy_uuid("123e4567-e89b-12d3-a456-426614174000".into()));
        assert!(policy_uuid(" 123E4567-E89B-12D3-A456-426614174000 ".into()));
        assert!(!policy_uuid("123e4567e89b12d3a456426614174000".into()));
        assert!(!policy_uuid("g23e4567-e89b-12d3-a456-426614174000".into()));
        assert!(!policy_uuid("123e4567-e89b-12d3-a456-42661417400-".into()));
    }

    #[test]
    fn reputation_weights_follow_the_reason() {
        for (reason, expected) in [("SQLi probe", 40), ("xss", 30), ("scanner", 20), ("other", 10)] {
            assert_eq!(policy_reputation(0, 0, vec![], reason.into()).0, expected, "{reason}");
        }
    }

    #[test]
    fn reputation_score_is_capped_at_100() {
        assert_eq!(policy_reputation(90, 0, vec![], "sqli".into()).0, 100);
    }

    #[test]
    fn reputation_deduplicates_reasons_and_names_blank_ones() {
        let (_, offenses, reasons, _, _) = policy_reputation(
            0,
            4,
            vec!["Scanner".into(), " scanner ".into(), "".into()],
            "   ".into(),
        );
        assert_eq!(offenses, 5);
        assert_eq!(reasons, vec!["Scanner".to_string(), "unknown".to_string()]);
    }

    #[test]
    fn reputation_block_duration_escalates() {
        let block = |score, offenses| {
            let (_, _, _, blocked, duration) = policy_reputation(score, offenses, vec![], "other".into());
            (blocked, duration)
        };
        assert_eq!(block(40, 0), (false, None));
        assert_eq!(block(50, 0), (true, Some(900)));
        assert_eq!(block(50, 1), (true, Some(3600)));
        assert_eq!(block(50, 2), (true, Some(86400)));
        assert_eq!(block(70, 0), (true, Some(86400)));
    }

    #[test]
    fn keyword_blocks_probe_paths_only_when_the_route_is_missing() {
        assert!(keyword("/.env", false, &[], &[]).0.unwrap().contains("probe path"));
        assert_eq!(keyword("/.env", true, &[], &[]).0, None);
    }

    #[test]
    fn keyword_leaves_normal_paths_alone() {
        assert_eq!(keyword("/docs/getting-started", true, &["shell"], &[]).0, None);
    }

    #[test]
    fn keyword_flags_inherently_suspicious_missing_paths() {
        let reason = keyword("/hackme", false, &[], &[]).0.unwrap();
        assert_eq!(reason, "Keyword block: Inherently suspicious: hackme");
    }

    #[test]
    fn keyword_only_blocks_existing_routes_with_strong_evidence() {
        assert_eq!(keyword("/shell/page", true, &["shell"], &["shell"]).0, None);
        let reason = keyword("/shell/<script", true, &["shell"], &["shell"]).0.unwrap();
        assert_eq!(reason, "Keyword block: Learned keyword: shell");
    }

    #[test]
    fn keyword_segments_are_lowercase_and_longer_than_three_chars() {
        assert_eq!(keyword("/API/a/LongName", true, &[], &[]).2, vec!["longname".to_string()]);
    }

    #[test]
    fn keyword_learns_only_from_missing_routes_with_context() {
        let learn = |exists: bool, contexts: &[&str], legitimate: &[&str]| {
            policy_keyword(
                "/wpconfig/page".into(),
                set(&[]),
                exists,
                true,
                set(&[]),
                set(&[]),
                set(legitimate),
                set(&[]),
                vec![],
                set(&[]),
                set(contexts),
            )
            .1
        };
        assert_eq!(learn(false, &["wpconfig"], &[]), vec!["wpconfig".to_string()]);
        assert!(learn(true, &["wpconfig"], &[]).is_empty());
        assert!(learn(false, &[], &[]).is_empty());
        assert!(learn(false, &["wpconfig"], &["wpconfig"]).is_empty());
    }

    #[test]
    fn keyword_safe_prefixes_skip_learned_keywords() {
        let check = |prefixes: Vec<String>| {
            policy_keyword(
                "/admin/panel".into(),
                set(&[]),
                false,
                false,
                set(&["panel"]),
                set(&[]),
                set(&[]),
                set(&[]),
                prefixes,
                set(&[]),
                set(&[]),
            )
            .0
        };
        assert!(check(vec![]).is_some());
        assert_eq!(check(vec!["admin".to_string()]), None);
    }

    #[test]
    fn sql_checks_use_the_tokenizer() {
        for (text, rule) in [
            ("1 UNION ALL SELECT password", "sql_breakout_numeric"),
            ("x' OR 1=1--", "sql_breakout_single_quote"),
            ("x'--", "sql_breakout_single_quote"),
            ("1; DROP TABLE users", "sql_breakout_numeric"),
            ("id=1 OR 1=1&page=2", "sql_breakout_numeric"),
            ("1 /*!50000UNION*/ SELECT 1", "sql_breakout_numeric"),
        ] {
            assert_eq!(sql_signature(text).as_deref(), Some(rule), "{text}");
        }
    }

    #[test]
    fn sql_signatures_ignore_normal_text() {
        for text in ["", "credit union selection", "it's a nice day", "select a seat", "\"Item\" #3"] {
            assert_eq!(sql_signature(text), None, "{text}");
        }
    }
}
