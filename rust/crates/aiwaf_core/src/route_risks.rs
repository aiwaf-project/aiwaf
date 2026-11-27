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
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const MAX_VALUE_CHARS: usize = 4096;
const MAX_ITEMS: usize = 512;
const MAX_DEPTH: usize = 8;
const COMMANDS: &str = "cat|curl|wget|nc|ncat|bash|sh|zsh|cmd|powershell|rm|id|whoami|uname|ping|ls|echo|sleep|nslookup|python|perl";
const NOSQL: &str = "where|ne|eq|gt|gte|lt|lte|in|nin|regex|exists|or|and|nor|not|expr";

/// The operations a saved finding can name.
pub const SINKS: &[&str] = &[
    "sql", "nosql", "command", "path", "ssrf", "code", "template", "deserialization", "xss", "redirect", "ldap", "xpath",
    "xxe", "redos", "prototype",
];

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
            // Markup that runs script, event handlers and script URLs.
            rule("xss", "xss_markup", r#"(?i)<\s*/?\s*(?:script|iframe|object|embed|svg|math|img|body|link|meta|style|base|form|input|video|audio|details)(?-u:\b)"#.into()),
            rule("xss", "xss_event_handler", r#"(?i)[\s"'/]on[a-z]{3,}\s*="#.into()),
            rule("xss", "xss_script_url", r#"(?i)(?:java|vb)script\s*:|data\s*:\s*text/html"#.into()),
            // Redirect targets that leave the site by a trick: // and its backslash variants, script and
            // data URLs, user@host URLs and https:host. Plain absolute URLs pass: the site's own host isn't known here.
            rule("redirect", "redirect_external", r#"(?i)^\s*(?:[/\\]{2}|/\s*\\|(?:java|vb)script\s*:|data\s*:|https?:(?:[^/\\]|[/\\][^/\\])|https?:[/\\]{2}[^/?#\s]*@)"#.into()),
            // Filter syntax that closes the expected value and starts another condition.
            rule("ldap", "ldap_filter", r#"(?i)\)\(|\(\s*[|&!]\s*\(|^\s*\*\s*\)|\x00"#.into()),
            // A quote that closes the expected literal, and the classic always-true condition.
            rule("xpath", "xpath_quote_break", r#"(?i)['"]\s*(?:or|and)\s+['"]?\w+['"]?\s*!?=|['"]\s*\]|['"]\s*\|\s*/"#.into()),
            rule("xpath", "xpath_tautology", r#"(?i)(?-u:\b)or\s+['"]?1['"]?\s*=\s*['"]?1"#.into()),
            // A document type or entity declaration, which external entities need.
            rule("xxe", "xxe_entity", r#"(?i)<!\s*(?:DOCTYPE|ENTITY)"#.into()),
            // A repeated group that repeats, as in (a+)+, which can take exponential time.
            rule("redos", "redos_nested_quantifier", r#"(?i)\([^()]*[+*][^()]*\)\s*(?:[+*]|\{\d)"#.into()),
            // Keys that reach an object's prototype.
            rule("prototype", "prototype_key", r#"(?i)__proto__|(?-u:\b)constructor\s*(?:[.\]"']\s*)+\[?\s*["']?prototype|^\s*(?:constructor|prototype)\s*$"#.into()),
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
    /// The inputs `init` found the route reads, with their types, and whether that list is
    /// complete. None when the audit couldn't tell.
    inputs: Option<HashMap<String, String>>,
    complete: bool,
    /// Each route parameter's type from the route's own syntax, in the order of compiled.names.
    path_types: Vec<Option<&'static str>>,
    /// The names each method reads, when init could tell them apart (`inputs_by_method`); a
    /// method without its own list reads the route's whole list.
    by_method: HashMap<String, HashSet<String>>,
    /// The keys read inside object-valued inputs, as {"user": ["email", "name"]}
    /// (`inputs_nested`), for inputs whose every use reads a constant key.
    nested: HashMap<String, HashSet<String>>,
    /// A method's own type for a name, where methods on the path differ, as POST's integer mode
    /// next to GET's text mode (`inputs_types_by_method`). `inputs` keeps the type every method
    /// agrees on, which is what an engine without this reads.
    method_types: HashMap<String, HashMap<String, String>>,
    /// A method's type for a name in one location, where the query and body differ, as a query
    /// limit that's an integer next to a JSON limit that's text (`inputs_types_by_location`,
    /// method to "query" or "body" to name to type).
    location_types: HashMap<String, HashMap<String, HashMap<String, String>>>,
}

impl RiskRoute {
    /// The type a value sent in this location ("query" or "body") checks against: the method's
    /// type for that location, else the method's own, else the route's.
    fn kind_at(&self, method: &str, location: &str, name: &str) -> Option<&str> {
        let upper = method.to_ascii_uppercase();
        let own = self.location_types.get(&upper).or_else(|| if upper == "HEAD" { self.location_types.get("GET") } else { None });
        own.and_then(|locations| locations.get(location)).and_then(|types| types.get(name)).map(String::as_str)
            .or_else(|| self.kind(method, name))
    }

    /// The type a request with this method checks a name against: the method's own, else the route's.
    fn kind(&self, method: &str, name: &str) -> Option<&str> {
        let method = method.to_ascii_uppercase();
        let own = self.method_types.get(&method).or_else(|| if method == "HEAD" { self.method_types.get("GET") } else { None });
        own.and_then(|types| types.get(name))
            .or_else(|| self.inputs.as_ref().and_then(|inputs| inputs.get(name)))
            .map(String::as_str)
    }

    /// The names a request with this method may send under strict inputs.
    fn reads(&self, method: &str, name: &str) -> bool {
        let method = method.to_ascii_uppercase();
        let own = self.by_method.get(&method).or_else(|| if method == "HEAD" { self.by_method.get("GET") } else { None });
        match own {
            Some(names) => names.contains(name),
            None => self.inputs.as_ref().is_some_and(|inputs| inputs.contains_key(name)),
        }
    }
}

/// What a route's saved inputs say about a request, from `RouteRisks::review`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Review {
    /// A typed input, as an int, sent with a value that isn't one.
    Mismatch { route: String, input: String, kind: String },
    /// The general check's finding comes only from inputs the route's schema clears: typed
    /// inputs of their type, text that reaches no operation, and inputs the route never reads.
    Relax { route: String, rule: String },
    /// Strict inputs: a query or body name a route with a complete input list never reads.
    Unexpected { route: String, input: String },
}

/// Names strict inputs always allows: CSRF tokens, read by middleware rather than the handler,
/// and tracking and cache-busting parameters that links and clients add. A trailing * matches
/// any rest.
/// Strict inputs' settings from the manifest's `strict_inputs`: names always allowed, and path
/// prefixes it skips.
struct Strict {
    allow: Vec<String>,
    exempt: Vec<String>,
}

/// A form name's key inside its input, as role in user[role]; None for a plain name.
fn bracket_key(raw: &str) -> Option<(String, String)> {
    let key = decode(raw.split('=').next().unwrap_or_default().as_bytes());
    let (top, rest) = key.split_once('[')?;
    let inner = rest.split(']').next()?;
    (!top.is_empty() && !inner.is_empty()).then(|| (top.to_owned(), inner.to_owned()))
}

pub const STRICT_ALLOWED: &[&str] = &[
    "csrfmiddlewaretoken", "csrf_token", "_csrf", "_token", "authenticity_token", "utm_*", "fbclid", "gclid", "msclkid", "_",
];
const MAX_LOGGED_NAME: usize = 64;

fn allowed(patterns: &[String], name: &str) -> bool {
    patterns.iter().any(|pattern| match pattern.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix),
        None => pattern == name,
    })
}

impl Review {
    /// The log line for this review: names and rule names, never request values.
    pub fn event(&self, mode: &str, method: &str) -> Value {
        match self {
            Review::Mismatch { route, input, kind } => serde_json::json!({
                "event": "aiwaf.route_schema", "mode": mode, "method": method, "route": route,
                "action": "mismatch", "input": input, "type": kind,
            }),
            Review::Relax { route, rule } => serde_json::json!({
                "event": "aiwaf.route_schema", "mode": mode, "method": method, "route": route,
                "action": "relax", "rule": rule,
            }),
            // The name comes from the request, so it's cut short; serde escapes it.
            Review::Unexpected { route, input } => serde_json::json!({
                "event": "aiwaf.route_schema", "mode": mode, "method": method, "route": route,
                "action": "unexpected", "input": input.chars().take(MAX_LOGGED_NAME).collect::<String>(),
            }),
        }
    }
}

/// Where the payload check's finding came from: the query or body, a header, or the URL path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Payload,
    Headers,
    Path,
}

/// The type a route segment's own syntax gives, as in <int:id> or {id:uuid}.
fn segment_type(segment: &str) -> Option<&'static str> {
    let converter = segment
        .strip_prefix('<')
        .and_then(|rest| rest.split_once(':'))
        .map(|(kind, _)| kind)
        .or_else(|| segment.strip_prefix('{').and_then(|rest| rest.strip_suffix('}')).and_then(|rest| rest.split_once(':')).map(|(_, kind)| kind))?;
    match converter {
        "int" => Some("int"),
        "float" => Some("float"),
        "uuid" => Some("uuid"),
        _ => None,
    }
}

/// The types an input can be saved with. Anything else is text.
pub const INPUT_TYPES: &[&str] = &["int", "float", "bool", "uuid", "text"];

/// True when a value fits a type. Values that can't carry text, as JSON numbers, always fit.
/// How deep strict inputs follow keys inside a JSON body.
const MAX_NESTED_DEPTH: usize = 6;

/// The first key below path that init knows the keys around and that isn't one of them: an
/// object's keys at "customer.address", each array item's at "lines[]". The first level ("user")
/// is checked by the caller; this goes deeper. Only paths init saved are checked.
fn deep_unexpected(nested: &HashMap<String, HashSet<String>>, allow: &[String], path: &str, value: &Value, depth: usize) -> Option<String> {
    if depth >= MAX_NESTED_DEPTH {
        return None;
    }
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let child_path = format!("{path}.{key}");
                if depth > 0 {
                    if let Some(keys) = nested.get(path) {
                        if !keys.contains(key) && !allowed(allow, &child_path) {
                            return Some(child_path);
                        }
                    }
                }
                if let Some(found) = deep_unexpected(nested, allow, &child_path, child, depth + 1) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(items) => {
            let item_path = format!("{path}[]");
            for item in items.iter().take(256) {
                if let (Some(keys), Some(object)) = (nested.get(&item_path), item.as_object()) {
                    if let Some(key) = object.keys().find(|key| !keys.contains(*key) && !allowed(allow, &format!("{item_path}.{key}"))) {
                        return Some(format!("{item_path}.{key}"));
                    }
                }
                if let Some(object) = item.as_object() {
                    for (key, child) in object {
                        if let Some(found) = deep_unexpected(nested, allow, &format!("{item_path}.{key}"), child, depth + 1) {
                            return Some(found);
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn fits(kind: &str, value: &Value) -> bool {
    static INT: OnceLock<Regex> = OnceLock::new();
    static FLOAT: OnceLock<Regex> = OnceLock::new();
    static UUID: OnceLock<Regex> = OnceLock::new();
    match value {
        Value::String(text) => {
            let text = text.trim();
            text.is_empty()
                || match kind {
                    "int" => INT.get_or_init(|| Regex::new(r"^[+-]?[0-9]{1,20}$").unwrap()).is_match(text),
                    "float" => FLOAT
                        .get_or_init(|| Regex::new(r"(?i)^[+-]?(?:[0-9]{1,40}\.?[0-9]{0,40}|\.[0-9]{1,40})(?:e[+-]?[0-9]{1,4})?$").unwrap())
                        .is_match(text),
                    "bool" => ["true", "false", "1", "0", "on", "off", "yes", "no", "t", "f", "y", "n"]
                        .contains(&text.to_ascii_lowercase().as_str()),
                    "uuid" => UUID
                        .get_or_init(|| {
                            Regex::new(r"(?i)^(?:urn:uuid:)?\{?[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}\}?$").unwrap()
                        })
                        .is_match(text),
                    _ => true,
                }
        }
        // A list of values, as in ?id=1&id=2 or {"ids": [1, 2]}.
        Value::Array(items) => items.iter().take(MAX_ITEMS).all(|item| !item.is_object() && fits(kind, item)),
        Value::Object(_) => kind == "text",
        _ => true,
    }
}

/// A request's top-level fields, as (name, value, raw form pair or JSON key), and whether its
/// body has a shape the review can read: none, a form or a JSON object.
struct Fields {
    query: Vec<(String, Value, String)>,
    body: Option<BodyFields>,
}

enum BodyFields {
    Empty,
    Form(Vec<(String, Value, String)>),
    Json(serde_json::Map<String, Value>),
    /// A multipart form's parts, with text parts' values; a file part's value is null.
    Multipart(Vec<(String, Value, String)>),
}

fn form_fields(text: &str) -> Vec<(String, Value, String)> {
    text.split('&')
        .filter(|part| !part.is_empty())
        .take(MAX_ITEMS)
        .map(|part| {
            let (key, value) = part.split_once('=').unwrap_or((part, ""));
            (input_name(&decode(key.as_bytes())), Value::String(decode(value.as_bytes())), part.to_owned())
        })
        .collect()
}

fn fields(query: &str, body: &[u8]) -> Fields {
    if let Some(parts) = crate::payload::multipart_parts(body) {
        let items = parts
            .into_iter()
            .take(MAX_ITEMS)
            .filter(|part| !part.name.is_empty())
            .map(|part| {
                let value = match part.filename {
                    Some(_) => Value::Null,
                    None => Value::String(String::from_utf8_lossy(&part.content[..part.content.len().min(MAX_VALUE_CHARS)]).into_owned()),
                };
                (input_name(&part.name), value, part.name)
            })
            .collect();
        return Fields { query: form_fields(query), body: Some(BodyFields::Multipart(items)) };
    }
    let body = match std::str::from_utf8(body) {
        Ok(text) if text.trim().is_empty() => Some(BodyFields::Empty),
        Ok(text) if text.trim_start().starts_with('{') => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(map)) => Some(BodyFields::Json(map)),
            _ => None,
        },
        Ok(text) if text.contains('=') && !text.trim_start().starts_with(['[', '<', '-']) => Some(BodyFields::Form(form_fields(text))),
        _ => None,
    };
    Fields { query: form_fields(query), body }
}

fn high_confidence(value: &str, budget: usize) -> Option<String> {
    crate::payload::inspect_sql(b"", value.as_bytes(), budget, false)
        .map(|(rule, _)| rule)
        .filter(|rule| !RELAXED_RULES.contains(&rule.as_str()))
}

/// The general check's rules a cleared input only logs: the ones that fire on prose and code
/// snippets. Quote breakouts, as in ' OR 1=1--, and every remote code execution rule still block.
pub const RELAXED_RULES: &[&str] = &["sql_function", "sql_comment", "sql_breakout_numeric"];

/// Saved risks for one manifest, compiled for lookup by request path.
pub struct RouteRisks {
    routes: Vec<RiskRoute>,
    /// Strict inputs' settings, when the manifest's `strict_inputs` turns it on.
    strict: Option<Strict>,
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
                let saved = entry.get("injection_risks").and_then(Value::as_array).cloned().unwrap_or_default();
                let inputs = entry.get("inputs").and_then(Value::as_object).map(|inputs| {
                    inputs
                        .iter()
                        .map(|(name, kind)| {
                            let kind = kind.as_str().filter(|kind| INPUT_TYPES.contains(kind)).unwrap_or("text");
                            (name.clone(), kind.to_owned())
                        })
                        .collect::<HashMap<_, _>>()
                });
                let complete = inputs.is_some() && entry.get("inputs_complete").and_then(Value::as_bool) == Some(true);
                let risks: Vec<Risk> = saved
                    .iter()
                    .filter(|risk| risk.is_object())
                    .map(|risk| Risk {
                        sink: text_field(risk, "sink"),
                        input: text_field(risk, "input"),
                        confidence: text_field(risk, "confidence"),
                    })
                    .collect();
                if !risks.is_empty() || inputs.is_some() {
                    let path_types = path
                        .split('/')
                        .filter(|segment| !segment.is_empty())
                        .filter(|segment| segment.starts_with(['<', '{', ':', '*']))
                        .map(segment_type)
                        .collect();
                    let nested = entry
                        .get("inputs_nested")
                        .and_then(Value::as_object)
                        .map(|keys| {
                            keys.iter()
                                .filter_map(|(name, keys)| {
                                    Some((name.clone(), keys.as_array()?.iter().filter_map(Value::as_str).map(str::to_owned).collect()))
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let by_method = entry
                        .get("inputs_by_method")
                        .and_then(Value::as_object)
                        .map(|methods| {
                            methods
                                .iter()
                                .filter_map(|(method, names)| {
                                    let names = names.as_array()?.iter().filter_map(Value::as_str).map(str::to_owned).collect();
                                    Some((method.to_ascii_uppercase(), names))
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let method_types = entry
                        .get("inputs_types_by_method")
                        .and_then(Value::as_object)
                        .map(|methods| {
                            methods
                                .iter()
                                .filter_map(|(method, types)| {
                                    let types = types
                                        .as_object()?
                                        .iter()
                                        .filter_map(|(name, kind)| {
                                            let kind = kind.as_str().filter(|kind| INPUT_TYPES.contains(kind))?;
                                            Some((name.clone(), kind.to_owned()))
                                        })
                                        .collect();
                                    Some((method.to_ascii_uppercase(), types))
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let types_of = |value: &Value| -> HashMap<String, String> {
                        value.as_object().map(|types| {
                            types.iter().filter_map(|(name, kind)| {
                                Some((name.clone(), kind.as_str().filter(|kind| INPUT_TYPES.contains(kind))?.to_owned()))
                            }).collect()
                        }).unwrap_or_default()
                    };
                    let location_types = entry
                        .get("inputs_types_by_location")
                        .and_then(Value::as_object)
                        .map(|methods| {
                            methods.iter().filter_map(|(method, locations)| {
                                let locations = locations.as_object()?.iter()
                                    .filter(|(location, _)| ["query", "body"].contains(&location.as_str()))
                                    .map(|(location, types)| (location.clone(), types_of(types)))
                                    .collect();
                                Some((method.to_ascii_uppercase(), locations))
                            }).collect()
                        })
                        .unwrap_or_default();
                    routes.push(RiskRoute {
                        path: path.clone(),
                        compiled: compile_route(path),
                        risks,
                        inputs,
                        complete,
                        path_types,
                        by_method,
                        nested,
                        method_types,
                        location_types,
                    });
                }
            }
        }
        // Exact routes win over routes with parameters.
        routes.sort_by(|a, b| {
            a.compiled.names.len().cmp(&b.compiled.names.len()).then(b.path.len().cmp(&a.path.len()))
        });
        // Hosts turn strict inputs on by adding {"strict_inputs": {"allow": [...]}}, the names
        // to allow besides STRICT_ALLOWED, to the manifest they pass.
        let strings = |value: &Value, key: &str| -> Vec<String> {
            value.get(key).and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect()
        };
        let strict = manifest.get("strict_inputs").filter(|value| value.is_object()).map(|value| {
            let mut allow: Vec<String> = STRICT_ALLOWED.iter().map(|name| (*name).to_owned()).collect();
            allow.extend(strings(value, "allow"));
            let exempt = strings(value, "exempt").into_iter().map(|prefix| prefix.trim_end_matches('*').to_owned()).filter(|prefix| !prefix.is_empty()).collect();
            Strict { allow, exempt }
        });
        Self { routes, strict }
    }

    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// Whether strict inputs is on.
    pub fn strict(&self) -> bool {
        self.strict.is_some()
    }

    /// Whether this engine skips strict inputs' exempt paths, and checks multipart names and
    /// keys inside inputs: hosts give an older one no exempt list it would ignore.
    pub fn strict_exempt(&self) -> bool {
        true
    }

    fn route(&self, path: &str) -> Option<&RiskRoute> {
        let path = if path.is_empty() { "/" } else { path };
        self.routes.iter().find(|route| route.compiled.pattern.is_match(path))
    }

    /// Reviews the payload check's finding, by where it came from, or a request with none. A
    /// header's finding always stands. A path's is reviewed by `review_path`.
    pub fn review_found(&self, path: &str, query: &str, body: &[u8], budget: usize, found: Option<(&str, Source)>) -> Option<Review> {
        self.review_found_method("", path, query, body, budget, found)
    }

    /// As `review_found`, for a request with this method: strict inputs check the method's own
    /// list when the manifest has one.
    pub fn review_found_method(
        &self,
        method: &str,
        path: &str,
        query: &str,
        body: &[u8],
        budget: usize,
        found: Option<(&str, Source)>,
    ) -> Option<Review> {
        match found {
            None => self.review_method(method, path, query, body, budget, None),
            Some((rule, Source::Payload)) => self.review_method(method, path, query, body, budget, Some(rule)),
            Some((_, Source::Headers)) => None,
            Some((rule, Source::Path)) => self.review_path(path, budget, rule),
        }
    }

    /// Reviews a finding from the URL path, as in /posts/use-sleep(5)-in-queries. The route's
    /// static segments are code, so only its parameters can carry input: typed ones must fit
    /// their type, and text ones that reach no operation are cleared, still getting the rules
    /// that aren't in `RELAXED_RULES`. `Relax` when the path is clean without them.
    pub fn review_path(&self, path: &str, budget: usize, found: &str) -> Option<Review> {
        let route = self.route(path)?;
        let inputs = route.inputs.as_ref()?;
        let path = if path.is_empty() { "/" } else { path };
        let captures = route.compiled.pattern.captures(path)?;
        let relaxable = RELAXED_RULES.contains(&found) && !route.risks.iter().any(|risk| risk.input.is_empty());
        let mut rebuilt = String::with_capacity(path.len());
        let mut last = 0;
        for (index, name) in route.compiled.names.iter().enumerate() {
            let Some(value) = captures.get(index + 1) else { continue };
            let kind = route.path_types.get(index).copied().flatten().or_else(|| inputs.get(name).map(String::as_str)).unwrap_or("text");
            let text = decode(value.as_str().as_bytes());
            if kind != "text" {
                if !fits(kind, &Value::String(text)) {
                    return Some(Review::Mismatch { route: route.path.clone(), input: name.clone(), kind: kind.to_owned() });
                }
            } else if name.is_empty() || route.risks.iter().any(|risk| &risk.input == name) {
                continue;
            } else if high_confidence(&text, budget).is_some() {
                return None;
            }
            rebuilt.push_str(&path[last..value.start()]);
            rebuilt.push('1');
            last = value.end();
        }
        rebuilt.push_str(&path[last..]);
        if !relaxable {
            return None;
        }
        match crate::payload::inspect_sql_path(&rebuilt, budget) {
            None => Some(Review::Relax { route: route.path.clone(), rule: found.to_owned() }),
            Some(_) => None,
        }
    }

    /// Reviews a request against the inputs `init` saved for its route, after the general check.
    ///
    /// A typed input sent with a value that isn't of its type is a `Mismatch`. When `found`, the
    /// general check's finding, comes from the query or body, the request is checked again
    /// without the inputs the schema clears, which still get the rules that aren't in
    /// `RELAXED_RULES`. `Relax` when nothing else is found. None, keeping the finding, whenever
    /// the route's inputs aren't known in full: an incomplete list, an operation reached by an
    /// input the audit couldn't name, or a body that isn't a form or JSON object.
    pub fn review(&self, path: &str, query: &str, body: &[u8], budget: usize, found: Option<&str>) -> Option<Review> {
        self.review_method("", path, query, body, budget, found)
    }

    /// As `review`, for a request with this method ("" for any): strict inputs check the names
    /// that method reads when the manifest saved them (`inputs_by_method`).
    pub fn review_method(&self, method: &str, path: &str, query: &str, body: &[u8], budget: usize, found: Option<&str>) -> Option<Review> {
        let route = self.route(path)?;
        let inputs = route.inputs.as_ref()?;
        let fields = fields(query, body);
        let mut all: Vec<(&String, &Value)> = fields.query.iter().map(|(name, value, _)| (name, value)).collect();
        let in_query = all.len();
        match &fields.body {
            Some(BodyFields::Form(items) | BodyFields::Multipart(items)) => all.extend(items.iter().map(|(name, value, _)| (name, value))),
            Some(BodyFields::Json(map)) => all.extend(map.iter()),
            _ => (),
        }
        for (index, (name, value)) in all.iter().enumerate() {
            // Each value against its own location's type: the query's, or the body's.
            let location = if index < in_query { "query" } else { "body" };
            if let Some(kind) = route.kind_at(method, location, name).filter(|kind| *kind != "text") {
                if !fits(kind, value) {
                    // The name comes from the manifest, because request names are untrusted.
                    let input = inputs
                        .get_key_value(*name)
                        .or_else(|| route.method_types.values().find_map(|types| types.get_key_value(*name)))
                        .or_else(|| route.location_types.values().flat_map(HashMap::values).find_map(|types| types.get_key_value(*name)))
                        .map(|(key, _)| key.clone())
                        .unwrap_or_default();
                    return Some(Review::Mismatch { route: route.path.clone(), input, kind: kind.to_owned() });
                }
            }
        }
        // Strict inputs: on a route whose list is complete, a name it never reads. Only the query
        // and a form or JSON object body are named; another body's names aren't known.
        let request_path = if path.is_empty() { "/" } else { path };
        if let Some(strict) = self
            .strict
            .as_ref()
            .filter(|strict| route.complete && !strict.exempt.iter().any(|prefix| request_path.starts_with(prefix.as_str())))
        {
            let allow = &strict.allow;
            if let Some((name, _)) = all.iter().find(|(name, _)| !route.reads(method, name) && !allowed(allow, name)) {
                return Some(Review::Unexpected { route: route.path.clone(), input: (*name).clone() });
            }
            // Keys inside an object-valued input whose keys init knows, as user.role.
            let inner = |top: &str, key: &str| {
                route.nested.get(top).is_some_and(|keys| !keys.contains(key) && !allowed(allow, &format!("{top}.{key}")))
            };
            if let Some(BodyFields::Json(map)) = &fields.body {
                for (top, value) in map {
                    if let Some(key) = value.as_object().and_then(|object| object.keys().find(|key| inner(top, key))) {
                        return Some(Review::Unexpected { route: route.path.clone(), input: format!("{top}.{key}") });
                    }
                    // Deeper keys init knows, as customer.address.city, or lines[].sku in each item.
                    if let Some(input) = deep_unexpected(&route.nested, allow, top, value, 0) {
                        return Some(Review::Unexpected { route: route.path.clone(), input });
                    }
                }
            }
            let raw_pairs = fields.query.iter().map(|(_, _, raw)| raw.as_str()).chain(match &fields.body {
                Some(BodyFields::Form(items) | BodyFields::Multipart(items)) => items.iter().map(|(_, _, raw)| raw.as_str()).collect::<Vec<_>>(),
                _ => Vec::new(),
            });
            for raw in raw_pairs {
                if let Some((top, key)) = bracket_key(raw).filter(|(top, key)| inner(top, key)) {
                    return Some(Review::Unexpected { route: route.path.clone(), input: format!("{top}.{key}") });
                }
            }
        }
        let found = found?;
        if !RELAXED_RULES.contains(&found) || !route.complete || route.risks.iter().any(|risk| risk.input.is_empty()) {
            return None;
        }
        let body_fields = fields.body.as_ref()?;
        // A finding from headers or the path isn't the query's or body's to clear.
        crate::payload::inspect_sql(query.as_bytes(), body, budget, false)?;
        let reaches = |name: &str| route.risks.iter().any(|risk| risk.input == name);
        let cleared = |name: &str, value: &Value| -> Result<bool, ()> {
            match inputs.get(name).map(String::as_str) {
                Some(kind) if kind != "text" => Ok(true),
                Some(_) if reaches(name) => Ok(false),
                _ => {
                    let text = match value {
                        Value::String(text) => text.clone(),
                        other => other.to_string(),
                    };
                    if high_confidence(&text, budget).is_some() { Err(()) } else { Ok(true) }
                }
            }
        };
        let mut kept_query = Vec::new();
        for (name, value, raw) in &fields.query {
            if !cleared(name, value).ok()? {
                kept_query.push(raw.as_str());
            }
        }
        let kept_body = match body_fields {
            // A multipart body can't be rebuilt without the parts the schema clears.
            BodyFields::Multipart(_) => return None,
            BodyFields::Empty => String::new(),
            BodyFields::Form(items) => {
                let mut kept = Vec::new();
                for (name, value, raw) in items {
                    if !cleared(name, value).ok()? {
                        kept.push(raw.as_str());
                    }
                }
                kept.join("&")
            }
            BodyFields::Json(map) => {
                let mut kept = serde_json::Map::new();
                for (name, value) in map {
                    if !cleared(name, value).ok()? {
                        kept.insert(name.clone(), value.clone());
                    }
                }
                if kept.is_empty() { String::new() } else { Value::Object(kept).to_string() }
            }
        };
        match crate::payload::inspect_sql(kept_query.join("&").as_bytes(), kept_body.as_bytes(), budget, false) {
            None => Some(Review::Relax { route: route.path.clone(), rule: found.to_owned() }),
            Some(_) => None,
        }
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

    /// What a request comes to: the general check, then the route's schema review.
    #[test]
    fn a_method_checks_its_own_type_where_methods_differ() {
        let risks = RouteRisks::new(&json!({"routes": {
            "/plans": {"view": "p", "inputs": {"mode": "text", "weeks": "int"}, "inputs_complete": true,
                       "inputs_types_by_method": {"post": {"mode": "int"}, "GET": {"weeks": "text", "bad": "nonsense"}}}
        }}));
        let mismatch = |method: &str, query: &str, body: &str| match risks.review_method(method, "/plans", query, body.as_bytes(), 4096, None) {
            Some(Review::Mismatch { input, kind, .. }) => Some((input, kind)),
            _ => None,
        };
        // POST's own integer; GET keeps the route's text, and its own text for weeks.
        assert_eq!(mismatch("POST", "", r#"{"mode": "fast"}"#), Some(("mode".into(), "int".into())));
        assert_eq!(mismatch("POST", "", r#"{"mode": 3}"#), None);
        assert_eq!(mismatch("GET", "mode=fast&weeks=two", ""), None);
        // HEAD reads GET's; a method with no types of its own reads the route's.
        assert_eq!(mismatch("HEAD", "weeks=two", ""), None);
        assert_eq!(mismatch("DELETE", "weeks=two", ""), Some(("weeks".into(), "int".into())));
        assert_eq!(mismatch("DELETE", "mode=fast", ""), None);
        // Without the entry, as an older manifest, every method reads the route's types.
        let plain = RouteRisks::new(&json!({"routes": {"/plans": {"view": "p", "inputs": {"mode": "text"}, "inputs_complete": true}}}));
        assert!(plain.review_method("POST", "/plans", "", br#"{"mode": "fast"}"#, 4096, None).is_none());
    }

    #[test]
    fn each_value_checks_its_own_locations_type() {
        let risks = RouteRisks::new(&json!({"routes": {
            "/reports": {"view": "r", "inputs": {"limit": "text", "since": "int"}, "inputs_complete": true,
                         "inputs_types_by_location": {"POST": {"query": {"limit": "int"}, "body": {"limit": "text"}, "cookie": {"limit": "int"}}}}
        }}));
        let mismatch = |query: &str, body: &str| match risks.review_method("POST", "/reports", query, body.as_bytes(), 4096, None) {
            Some(Review::Mismatch { input, kind, .. }) => Some((input, kind)),
            _ => None,
        };
        // The valid pair: an integer query limit and a text JSON limit.
        assert_eq!(mismatch("limit=5", r#"{"limit": "all"}"#), None);
        // Each changed alone: only its own check fires.
        assert_eq!(mismatch("limit=all", r#"{"limit": "all"}"#), Some(("limit".into(), "int".into())));
        assert_eq!(mismatch("limit=5", r#"{"limit": 7}"#), None);
        // A name with no location type falls back to the route's; another method to the route's.
        assert_eq!(mismatch("since=soon", ""), Some(("since".into(), "int".into())));
        assert!(risks.review_method("GET", "/reports", "limit=all", b"", 4096, None).is_none());
        // A form body is the body location too.
        assert!(risks.review_method("POST", "/reports", "limit=5", b"limit=all", 4096, None).is_none());
    }

    #[test]
    fn strict_inputs_follow_keys_deeper_where_init_knows_them() {
        let risks = RouteRisks::new(&json!({
            "strict_inputs": {"allow": ["customer.address.note"]},
            "routes": {"/orders": {"view": "o", "inputs": {"customer": "text", "lines": "text"}, "inputs_complete": true,
                "inputs_nested": {"customer": ["address", "email"], "customer.address": ["city", "zip"], "lines[]": ["qty", "sku"]}}}
        }));
        let unexpected = |body: &str| match risks.review_method("POST", "/orders", "", body.as_bytes(), 4096, None) {
            Some(Review::Unexpected { input, .. }) => Some(input),
            _ => None,
        };
        let valid = r#"{"customer": {"email": "a@b.c", "address": {"city": "X", "zip": "1"}}, "lines": [{"sku": "s", "qty": 1}, {"sku": "t", "qty": 2}]}"#;
        assert_eq!(unexpected(valid), None);
        assert_eq!(unexpected(r#"{"customer": {"address": {"city": "X", "role": "admin"}}}"#), Some("customer.address.role".into()));
        assert_eq!(unexpected(r#"{"lines": [{"sku": "s"}, {"sku": "t", "price": 0}]}"#), Some("lines[].price".into()));
        assert_eq!(unexpected(r#"{"customer": {"address": {"note": "allowed by name"}}}"#), None);
        // A level init knows nothing about isn't checked: lines items' own objects, a string address.
        assert_eq!(unexpected(r#"{"lines": [{"sku": {"any": "thing"}}], "customer": {"address": "text"}}"#), None);
    }

    #[test]
    fn strict_inputs_flags_names_a_complete_route_never_reads() {
        let routes = json!({
            "/search": {"view": "s", "inputs": {"q": "text", "page": "int"}, "inputs_complete": true},
            "/open": {"view": "o", "inputs": {"q": "text"}, "inputs_complete": false},
        });
        let off = RouteRisks::new(&json!({"routes": routes}));
        assert!(!off.strict());
        assert_eq!(off.review("/search", "q=a&debug=1", b"", 4096, None), None);
        let on = RouteRisks::new(&json!({"routes": routes, "strict_inputs": {"allow": ["trace_*", 5]}}));
        assert!(on.strict());
        assert_eq!(
            on.review("/search", "q=a&debug=1", b"", 4096, None),
            Some(Review::Unexpected { route: "/search".into(), input: "debug".into() })
        );
        assert_eq!(
            on.review("/search", "q=a", br#"{"page": 2, "role": "admin"}"#, 4096, None),
            Some(Review::Unexpected { route: "/search".into(), input: "role".into() })
        );
        // A wrong type is reported first.
        assert!(matches!(on.review("/search", "page=x&debug=1", b"", 4096, None), Some(Review::Mismatch { .. })));
        for query in ["q=a&utm_source=x&_=1&csrf_token=t&trace_id=9&gclid=1", "page=2"] {
            assert_eq!(on.review("/search", query, b"", 4096, None), None, "{query}");
        }
        assert_eq!(on.review("/open", "debug=1", b"", 4096, None), None);
        assert_eq!(on.review("/search", "q=a", b"\x00binary", 4096, None), None);
        let event = Review::Unexpected { route: "/search".into(), input: "x".repeat(100) }.event("enforce", "GET");
        assert_eq!(event["action"], "unexpected");
        assert_eq!(event["input"].as_str().unwrap().len(), MAX_LOGGED_NAME);
        assert!(!RouteRisks::new(&json!({"routes": routes, "strict_inputs": true})).strict());
    }

    #[test]
    fn strict_inputs_check_each_methods_own_list() {
        let manifest = json!({"strict_inputs": {}, "routes": {"/posts": {
            "view": "p", "inputs": {"q": "text", "title": "text"}, "inputs_complete": true,
            "inputs_by_method": {"get": ["q"], "POST": ["title"], "PUT": "not a list"},
        }}});
        let risks = RouteRisks::new(&manifest);
        let unexpected = |input: &str| Some(Review::Unexpected { route: "/posts".into(), input: input.into() });
        assert_eq!(risks.review_method("GET", "/posts", "title=x", b"", 4096, None), unexpected("title"));
        assert_eq!(risks.review_method("HEAD", "/posts", "title=x", b"", 4096, None), unexpected("title"));
        assert_eq!(risks.review_method("get", "/posts", "q=x", b"", 4096, None), None);
        assert_eq!(risks.review_method("POST", "/posts", "", br#"{"q": 1}"#, 4096, None), unexpected("q"));
        // A method without its own list, or no method, reads the whole list.
        assert_eq!(risks.review_method("PATCH", "/posts", "q=1&title=2", b"", 4096, None), None);
        assert_eq!(risks.review("/posts", "q=1&title=2", b"", 4096, None), None);
        assert_eq!(risks.review_method("PUT", "/posts", "q=1", b"", 4096, None), None);
        assert_eq!(risks.review_found_method("GET", "/posts", "title=x", b"", 4096, None), unexpected("title"));
    }

    pub(crate) fn schema_outcome(risks: &RouteRisks, path: &str, query: &str, body: &[u8]) -> (String, String) {
        method_outcome(risks, "", path, query, body)
    }

    fn method_outcome(risks: &RouteRisks, method: &str, path: &str, query: &str, body: &[u8]) -> (String, String) {
        let found = crate::payload::inspect_sql(query.as_bytes(), body, 65536, false).map(|(rule, _)| rule);
        match risks.review_method(method, path, query, body, 65536, found.as_deref()) {
            Some(Review::Mismatch { input, .. }) => ("mismatch".into(), input),
            Some(Review::Unexpected { input, .. }) => ("unexpected".into(), input),
            Some(Review::Relax { .. }) => ("relax".into(), String::new()),
            None if found.is_some() => ("block".into(), String::new()),
            None => ("pass".into(), String::new()),
        }
    }

    #[test]
    fn shared_schema_cases() {
        let cases: Value = serde_json::from_str(include_str!("../../../../tests/fixtures/route_schema_cases.json")).unwrap();
        let risks = RouteRisks::new(&cases["manifest"]);
        for case in cases["cases"].as_array().unwrap() {
            let text = |key: &str| case[key].as_str().unwrap_or_default().to_owned();
            let outcome = schema_outcome(&risks, &text("path"), &text("query"), text("body").as_bytes());
            assert_eq!(outcome, (text("expect"), text("input")), "{case}");
        }
        for case in cases["path_cases"].as_array().unwrap() {
            let path = case["path"].as_str().unwrap();
            let found = crate::payload::inspect_sql_path(path, 65536).map(|(rule, _)| rule);
            let outcome = match risks.review_found(path, "", b"", 65536, found.as_deref().map(|rule| (rule, Source::Path))) {
                Some(Review::Mismatch { input, .. }) => ("mismatch".to_owned(), input),
                Some(Review::Unexpected { input, .. }) => ("unexpected".to_owned(), input),
                Some(Review::Relax { .. }) => ("relax".to_owned(), String::new()),
                None if found.is_some() => ("block".to_owned(), String::new()),
                None => ("pass".to_owned(), String::new()),
            };
            let input = case["input"].as_str().unwrap_or_default().to_owned();
            assert_eq!(outcome, (case["expect"].as_str().unwrap().to_owned(), input), "{case}");
        }
        let mut manifest = cases["manifest"].clone();
        manifest["strict_inputs"] = json!({"allow": cases["strict_cases"]["allow"], "exempt": cases["strict_cases"]["exempt"]});
        let strict = RouteRisks::new(&manifest);
        for case in cases["strict_cases"]["cases"].as_array().unwrap() {
            let text = |key: &str| case[key].as_str().unwrap_or_default().to_owned();
            let outcome = method_outcome(&strict, &text("method"), &text("path"), &text("query"), text("body").as_bytes());
            assert_eq!(outcome, (text("expect"), text("input")), "{case}");
        }
        // A header's finding always stands.
        assert_eq!(risks.review_found("/comments", "", b"", 65536, Some(("sql_function", Source::Headers))), None);
        let event = Review::Relax { route: "/x".into(), rule: "sql_function".into() }.event("monitor", "GET");
        assert_eq!(event["action"], "relax");
        // Routes with only saved inputs still give no route-specific events.
        assert!(risks.inspect("/comments", "body=a;id", b"").is_empty());
        assert!(fits("int", &json!(null)) && !fits("int", &json!({"a": 1})) && fits("text", &json!({"a": 1})));
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
