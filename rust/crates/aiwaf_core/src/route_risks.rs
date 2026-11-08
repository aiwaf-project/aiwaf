//! Route-specific checks built from audit findings saved in a route manifest.
//!
//! `init` saves which operation each route's input reaches (`injection_risks`). Requests to
//! those routes are checked for patterns that matter to that operation, in the parameter that
//! reaches it. Events carry names from the manifest and rule names only, never request values.
//!
//! This is the implementation Python uses through aiwaf-rust. Node.js and Java have their own
//! copies, and the cases in `tests/fixtures/route_risk_cases.json` and
//! `tests/fixtures/ordinary_text.json` run against every one.

use crate::payload::{JsonTree, decode, parse_json};
use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::OnceLock;

const MAX_VALUE_CHARS: usize = 4096;
const MAX_ITEMS: usize = 512;
const MAX_DEPTH: usize = 8;
const COMMANDS: &str = "cat|curl|wget|nc|ncat|bash|sh|zsh|cmd|powershell|rm|id|whoami|uname|ping|ls|echo|sleep|nslookup|python|perl";
const NOSQL: &str = "where|ne|eq|gt|gte|lt|lte|in|nin|regex|exists|or|and|nor|not|expr";

/// The operations a saved finding can name.
pub const SINKS: &[&str] = &["sql", "nosql", "command", "path", "ssrf", "code", "template", "deserialization"];

/// One match: the manifest route, the operation, the manifest's parameter name and the rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub route: String,
    pub sink: String,
    pub input: String,
    pub rule: String,
    pub confidence: String,
}

fn rules() -> &'static Vec<(&'static str, &'static str, Regex)> {
    static RULES: OnceLock<Vec<(&'static str, &'static str, Regex)>> = OnceLock::new();
    RULES.get_or_init(|| {
        let rule = |sink, name, pattern: String| (sink, name, Regex::new(&pattern).unwrap());
        vec![
            // Shell commands are lower-case. A chained command needs an argument that looks like one,
            // so "cat food" passes, and Markdown's spaced backticks, as in "The `id` field", aren't commands.
            rule("command", "command_chain", format!(r#"[;&|]\s*(?:{COMMANDS})(?:\s*$|\s*[;&|`)]|\s+\S*[-/.$~'"0-9:\\])"#)),
            rule("command", "command_substitution", format!(r"\$\(\s*(?:{COMMANDS})(?-u:\b)[^)]{{0,200}}\)|(?:^|\S)`\s*(?:{COMMANDS})(?-u:\b)[^`]{{0,200}}`")),
            rule("path", "path_traversal", r"(?:^|[/\\])\.\.(?:[/\\]|$)".into()),
            rule("path", "path_sensitive_location", r"(?i)^(?:/etc/|/proc/|/root/|/var/log/|[a-z]:[/\\]|\\\\)".into()),
            rule("path", "path_null_byte", r"\x00".into()),
            rule("ssrf", "ssrf_scheme", r"(?i)^\s*(?:file|gopher|dict|ftp|ldap|tftp|netdoc)://".into()),
            rule("ssrf", "ssrf_internal_host", r"(?i)^\s*(?:https?:)?//(?:[^/@?#\s]{0,128}@)?(?:localhost|127\.\d|0\.0\.0\.0|10\.\d|192\.168\.|169\.254\.|172\.(?:1[6-9]|2\d|3[01])\.|\[::1\]|metadata\.google\.internal)".into()),
            rule("nosql", "nosql_operator", format!(r#"(?i)^\$(?:{NOSQL})$|\[\$(?:{NOSQL})\]|\{{\s*"?\$(?:{NOSQL})(?-u:\b)"#)),
            rule("code", "code_expression", r"(?i)__import__|(?-u:\b)(?:eval|exec|compile)\s*\(|(?-u:\b)require\s*\(|(?-u:\b)process\.(?:env|exit|mainModule)|java\.lang\.|Runtime\.getRuntime|(?-u:\b)T\s*\(\s*java".into()),
            rule("template", "template_expression", r"(?i)\{\{.{0,200}\}\}|\{%.{0,200}%\}|\$\{.{0,200}\}|<%.{0,200}%>|#\{.{0,200}\}".into()),
            rule("deserialization", "deserialization_marker", r#"^rO0AB|^aced0005|^gASV|!!python/|!!javax\.|^O:\d+:"|__reduce__|_\$\$ND_FUNC\$\$_"#.into()),
        ]
    })
}

fn check(sink: &str, text: &str) -> Option<String> {
    if sink == "sql" {
        return crate::sql::classify_fields(text).map(str::to_owned);
    }
    rules()
        .iter()
        .find(|(rule_sink, _, pattern)| *rule_sink == sink && pattern.is_match(text))
        .map(|(_, name, _)| (*name).to_owned())
}

/// Returns the first rule for this operation that matches the value.
pub fn rule_for(sink: &str, value: &str) -> Option<String> {
    let text: String = value.chars().take(MAX_VALUE_CHARS).collect();
    check(sink, &text).or_else(|| if text.contains('%') { check(sink, &decode(text.as_bytes())) } else { None })
}

/// A route pattern compiled from Flask, Django, FastAPI, Express or Spring syntax.
pub struct CompiledRoute {
    pub pattern: Regex,
    pub names: Vec<String>,
}

/// Compiles a manifest route from any supported framework's syntax.
pub fn compile_route(path: &str) -> CompiledRoute {
    static SEGMENT: OnceLock<Regex> = OnceLock::new();
    static WILDCARD: OnceLock<Regex> = OnceLock::new();
    let segment_rule = SEGMENT.get_or_init(|| {
        Regex::new(r"^(?:<(?:[^:>]+:)?([^>]+)>|\{\*?([^:}]+)(?::[^}]*)?\}|:([^/?*]+)[?*]?)$").unwrap()
    });
    let wildcard = WILDCARD.get_or_init(|| {
        Regex::new(r"^(?:<path:[^>]+>|\{\*?[^:}]+:(?:path|\.\*|\.\+)\}|\{\*[^}]+\}|:[^/?*]+\*|\*\*)$").unwrap()
    });
    let mut pattern = String::from("^");
    let mut names = Vec::new();
    for segment in path.split('/').filter(|segment| !segment.is_empty()) {
        let name = segment_rule
            .captures(segment)
            .and_then(|captures| (1..=3).find_map(|i| captures.get(i)).map(|m| m.as_str().to_owned()));
        if wildcard.is_match(segment) {
            names.push(name.unwrap_or_default());
            pattern.push_str("/(.*)");
        } else if segment == "*" {
            names.push(String::new());
            pattern.push_str("/([^/]+)");
        } else if let Some(name) = name {
            names.push(name);
            pattern.push_str("/([^/]+)");
        } else {
            pattern.push('/');
            pattern.push_str(&regex::escape(segment));
        }
    }
    pattern.push_str("/?$");
    CompiledRoute { pattern: Regex::new(&pattern).unwrap(), names }
}

fn input_name(name: &str) -> String {
    name.split('[').next().unwrap_or_default().to_owned()
}

fn pairs(text: &str, items: &mut Vec<(String, String)>) {
    for part in text.split('&').filter(|part| !part.is_empty()) {
        if items.len() >= MAX_ITEMS {
            return;
        }
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        let key = decode(key.as_bytes());
        let name = input_name(&key);
        items.push((name.clone(), key));
        items.push((name, decode(value.as_bytes())));
    }
}

// Nested values also count as the top-level field that holds them, as in {"user": {"$ne": 1}}.
fn walk(value: &JsonTree, name: &str, root: &str, items: &mut Vec<(String, String)>, depth: usize) {
    if items.len() >= MAX_ITEMS || depth > MAX_DEPTH {
        return;
    }
    match value {
        JsonTree::Object(entries) => {
            for (key, item) in entries {
                let top = if root.is_empty() { key.as_str() } else { root };
                items.push((key.clone(), key.clone()));
                if top != key {
                    items.push((top.to_owned(), key.clone()));
                }
                walk(item, key, top, items, depth + 1);
            }
        }
        JsonTree::Array(entries) => {
            for item in entries {
                walk(item, name, root, items, depth + 1);
            }
        }
        JsonTree::Text(text) => {
            items.push((name.to_owned(), text.clone()));
            if !root.is_empty() && root != name {
                items.push((root.to_owned(), text.clone()));
            }
        }
        JsonTree::Scalar => {}
    }
}

/// Returns (name, value) pairs from a raw query string and a buffered body.
pub fn request_inputs(query: &str, body: &[u8]) -> Vec<(String, String)> {
    let mut items = Vec::new();
    pairs(query, &mut items);
    let text = String::from_utf8_lossy(body);
    if !text.is_empty() {
        let stripped = text.trim_start();
        if stripped.starts_with(['{', '[']) {
            match parse_json(stripped) {
                Ok(Some(tree)) => walk(&tree, "", "", &mut items, 0),
                _ => items.push((String::new(), text.into_owned())),
            }
        } else if text.contains('=') {
            pairs(&text, &mut items);
        } else {
            items.push((String::new(), text.into_owned()));
        }
    }
    items.truncate(MAX_ITEMS);
    items
}

struct Risk {
    sink: String,
    input: String,
    confidence: String,
}

struct RiskRoute {
    path: String,
    compiled: CompiledRoute,
    risks: Vec<Risk>,
}

/// Saved risks for one manifest, compiled for lookup by request path.
pub struct RouteRisks {
    routes: Vec<RiskRoute>,
}

fn text_field(value: &Value, key: &str) -> String {
    match value.get(key) {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

impl RouteRisks {
    pub fn new(manifest: &Value) -> Self {
        let mut routes = Vec::new();
        if let Some(entries) = manifest.get("routes").and_then(Value::as_object) {
            for (path, entry) in entries {
                let Some(saved) = entry.get("injection_risks").and_then(Value::as_array) else {
                    continue;
                };
                let risks: Vec<Risk> = saved
                    .iter()
                    .filter(|risk| risk.is_object())
                    .map(|risk| Risk {
                        sink: text_field(risk, "sink"),
                        input: text_field(risk, "input"),
                        confidence: text_field(risk, "confidence"),
                    })
                    .collect();
                if !risks.is_empty() {
                    routes.push(RiskRoute { path: path.clone(), compiled: compile_route(path), risks });
                }
            }
        }
        // Exact routes win over routes with parameters.
        routes.sort_by(|a, b| {
            a.compiled.names.len().cmp(&b.compiled.names.len()).then(b.path.len().cmp(&a.path.len()))
        });
        Self { routes }
    }

    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// Checks one request. Returns every distinct match on the first route the path matches.
    pub fn inspect(&self, path: &str, query: &str, body: &[u8]) -> Vec<Event> {
        let path = if path.is_empty() { "/" } else { path };
        for route in &self.routes {
            let Some(captures) = route.compiled.pattern.captures(path) else {
                continue;
            };
            let mut items = request_inputs(query, body);
            for (index, name) in route.compiled.names.iter().enumerate() {
                if let Some(value) = captures.get(index + 1).filter(|m| !m.as_str().is_empty()) {
                    items.push((name.clone(), value.as_str().to_owned()));
                }
            }
            let mut events = Vec::new();
            let mut seen = HashSet::new();
            for risk in &route.risks {
                for (name, value) in &items {
                    if !risk.input.is_empty() && name != &risk.input {
                        continue;
                    }
                    if let Some(rule) = rule_for(&risk.sink, value) {
                        if seen.insert((risk.sink.clone(), risk.input.clone(), rule.clone())) {
                            // The input name comes from the manifest, because request names are untrusted.
                            events.push(Event {
                                route: route.path.clone(),
                                sink: risk.sink.clone(),
                                input: risk.input.clone(),
                                rule,
                                confidence: risk.confidence.clone(),
                            });
                        }
                    }
                }
            }
            return events;
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cases() -> Value {
        serde_json::from_str(include_str!("../../../../tests/fixtures/route_risk_cases.json")).unwrap()
    }

    #[test]
    fn shared_rule_cases() {
        for case in cases()["rules"].as_array().unwrap() {
            let expected = case["rule"].as_str().map(str::to_owned);
            assert_eq!(rule_for(case["sink"].as_str().unwrap(), case["value"].as_str().unwrap()), expected, "{case}");
        }
    }

    #[test]
    fn shared_route_cases() {
        for case in cases()["routes"].as_array().unwrap() {
            let route = compile_route(case["route"].as_str().unwrap());
            let found = route.pattern.captures(case["path"].as_str().unwrap());
            if case["values"].is_null() {
                assert!(found.is_none(), "{case}");
                continue;
            }
            let found = found.unwrap_or_else(|| panic!("{case}"));
            let mut values = serde_json::Map::new();
            for (index, name) in route.names.iter().enumerate() {
                if !name.is_empty() {
                    values.insert(name.clone(), json!(found.get(index + 1).unwrap().as_str()));
                }
            }
            assert_eq!(Value::Object(values), case["values"], "{case}");
        }
    }

    #[test]
    fn ordinary_text_is_never_flagged() {
        let cases: Value = serde_json::from_str(include_str!("../../../../tests/fixtures/ordinary_text.json")).unwrap();
        let mut blocked = Vec::new();
        for value in cases["values"].as_array().unwrap() {
            for sink in SINKS {
                if let Some(rule) = rule_for(sink, value.as_str().unwrap()) {
                    blocked.push(format!("{sink}: {value} ({rule})"));
                }
            }
        }
        assert!(blocked.is_empty(), "{blocked:?}");
    }

    fn manifest(route: &str, risks: Value) -> Value {
        json!({"routes": {route: {"view": "v", "injection_risks": risks}, "/health": {"view": "h"}}})
    }

    #[test]
    fn only_the_named_input_is_checked_and_route_values_count() {
        let risks = RouteRisks::new(&manifest("/files/<path:name>", json!([{"sink": "path", "input": "name", "confidence": "medium"}])));
        assert!(risks.inspect("/files/ok", "other=../x", b"").is_empty());
        let events = risks.inspect("/files/../secret", "", b"");
        assert_eq!(events[0].rule, "path_traversal");
        assert_eq!(events[0].input, "name");
        assert_eq!(events[0].route, "/files/<path:name>");
    }

    #[test]
    fn bodies_and_nested_fields_are_inputs() {
        let risks = RouteRisks::new(&manifest("/s", json!([{"sink": "nosql", "input": "user"}, {"sink": "command", "input": ""}])));
        assert_eq!(risks.inspect("/s", "", br#"{"user": {"$ne": 1}}"#)[0].rule, "nosql_operator");
        assert_eq!(risks.inspect("/s", "user%5B%24ne%5D=1", b"")[0].rule, "nosql_operator");
        assert_eq!(risks.inspect("/s", "", b"a=%24%28id%29")[0].rule, "command_substitution");
        assert_eq!(risks.inspect("/s", "", b"$(id)")[0].rule, "command_substitution");
        assert_eq!(risks.inspect("/s", "", b"{broken $(id)")[0].rule, "command_substitution");
    }

    #[test]
    fn exact_routes_win_and_repeats_report_once() {
        let manifest = json!({"routes": {
            "/items/{id}": {"injection_risks": [{"sink": "path", "input": ""}]},
            "/items/new": {"injection_risks": [{"sink": "command", "input": ""}]}
        }});
        let risks = RouteRisks::new(&manifest);
        assert_eq!(risks.inspect("/items/new", "x=$(id)", b"")[0].route, "/items/new");
        let one = RouteRisks::new(&json!({"routes": {"/s": {"injection_risks": [{"sink": "path", "input": ""}]}}}));
        assert_eq!(one.inspect("/s", "a=../x&b=../y", b"").len(), 1);
    }

    #[test]
    fn routes_without_risks_and_bad_manifests_get_nothing() {
        let risks = RouteRisks::new(&manifest("/s", json!([{"sink": "path", "input": ""}])));
        assert!(risks.inspect("/health", "x=../y", b"").is_empty());
        assert!(risks.inspect("/other", "x=../y", b"").is_empty());
        assert!(RouteRisks::new(&json!(null)).is_empty());
        assert!(RouteRisks::new(&json!({"routes": "bad"})).is_empty());
        assert!(RouteRisks::new(&json!({"routes": {"/x": {"injection_risks": ["bad"]}}})).is_empty());
    }

    #[test]
    fn inputs_are_bounded() {
        let query = (0..5000).map(|i| format!("k{i}=v")).collect::<Vec<_>>().join("&");
        assert!(request_inputs(&query, b"").len() <= MAX_ITEMS);
        let deep = format!("{}\"x\"{}", "{\"a\":".repeat(20), "}".repeat(20));
        assert!(!request_inputs("", deep.as_bytes()).contains(&("a".into(), "x".into())));
        assert_eq!(rule_for("path", &format!("{}../x", "a".repeat(5000))), None);
    }
}
