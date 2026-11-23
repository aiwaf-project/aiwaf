//! Stateful, bounded Node/WASM request processing. Filesystem I/O belongs to Node.
use aiwaf_core::{
    HeaderLimits, IsolationForest, RecentEntryInput, analyze_recent_behavior, validate_headers_full,
    UserAgentLists, validate_headers_node, validate_headers_with_config_limits, validate_url,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};

pub const STAGES: [&str; 8] = [
    "geo_block",
    "ip_keyword_block",
    "rate_limit",
    "ai_anomaly",
    "honeypot",
    "uuid_tamper",
    "header_validation",
    "logging",
];
static NEXT_CONTEXT: AtomicU64 = AtomicU64::new(1);
/// Optional request facts this engine accepts, for `NativeRequestPipeline::supports`.
pub const FEATURES: [&str; 17] = [
    "rate_overrides",
    "route_schema",
    "strict_inputs",
    "strict_exempt",
    "sql_monitor",
    "uuid_resolver",
    "host_store",
    "rate_count",
    "header_limits",
    "login_prefixes",
    "keyword_exemptions",
    "post_only_suffixes",
    "method_headers",
    "header_policy",
    "user_agent_lists",
    "learn_context",
    "user_agent_verdict",
];

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Profile {
    enabled: HashSet<String>,
    exempt_paths: Vec<String>,
    exempt_ips: HashSet<String>,
    static_keywords: Vec<String>,
    dynamic_keywords: HashSet<String>,
    legitimate_keywords: Vec<String>,
    /// Keywords that never block, and path fragments that keep a request from keyword blocks,
    /// as the regular AIWAF_EXEMPT_KEYWORDS and AIWAF_ALLOWED_PATH_KEYWORDS (lowercased).
    exempt_keywords: Vec<String>,
    allowed_path_keywords: Vec<String>,
    learning: bool,
    geo_allow: HashSet<String>,
    geo_block: HashSet<String>,
    #[serde(rename = "blacklist", alias = "blocklist")]
    blocklist: HashMap<String, f64>,
    rate_window: f64,
    rate_max: usize,
    rate_flood: usize,
    rate_key_mode: String,
    min_form_time: f64,
    max_page_time: f64,
    skip_authenticated: bool,
    honeypot_field: String,
    /// Paths under these prefixes are login forms, which may be posted faster (lowercased paths).
    login_prefixes: Vec<String>,
    /// "node" learns keywords as Node.js's regular middleware does (anomalyDetector
    /// maybeLearnKeyword): every segment of a 404 that looks like a scan, from every request
    /// that wasn't blocked or exempt. Empty for the native rule.
    learn_context: String,
    learn_scan_patterns: Vec<String>,
    learn_static_keywords: Vec<String>,
    /// With a method policy: a GET to a path ending in one of these is refused (lowercased).
    post_only_suffixes: Vec<String>,
    allowed_methods: Vec<String>,
    uuid_prefix: String,
    uuid_window: f64,
    uuid_threshold: usize,
    anomaly_window: f64,
    required_headers: Vec<String>,
    /// Required headers for particular methods (uppercase); other methods use required_headers.
    required_headers_by_method: HashMap<String, Vec<String>>,
    header_min_score: i32,
    /// The header check's caps and blocked user agents (by substring), as the regular middleware's.
    header_max_bytes: usize,
    header_max_count: usize,
    header_max_user_agent: usize,
    header_max_accept: usize,
    blocked_user_agents: Vec<String>,
    /// Count header names without "HTTP_" toward header_max_bytes, as Node.js sends them.
    header_bytes_without_prefix: bool,
    /// "python" (the default) or "node": Node.js's regular header check, which skips static
    /// files (header_skip_extensions and header_skip_paths, matched on the lowercased path) and,
    /// with header_wasm_check, then runs aiwaf-wasm's own check as the regular middleware does.
    /// "node_ua" is what Node.js's regular middleware checks with AIWAF_HEADER_VALIDATION off:
    /// aiwaf-wasm's User-Agent check alone, with header_wasm_check.
    header_policy: String,
    header_skip_extensions: Vec<String>,
    header_skip_paths: Vec<String>,
    header_wasm_check: bool,
    /// The app's own User-Agent pattern lists for the "node" policy (JavaScript sources).
    suspicious_user_agents: Option<Vec<String>>,
    legitimate_bots: Option<Vec<String>>,
    sql_mode: String,
    max_bytes: usize,
    block_duration: f64,
    max_clients: usize,
    max_history: usize,
    max_pending: usize,
    /// "memory" keeps blocks and learned keywords in the engine. "host" takes them as request
    /// facts and returns blocks and keywords from finish for the host's stores.
    policy_store: String,
    /// What the inputs init saved for each route do, as AIWAF_ROUTE_SCHEMA_MODE: "enforce",
    /// "monitor" or "off". The route manifest is given in route_manifest.
    route_schema_mode: String,
    route_manifest: Value,
}
impl Default for Profile {
    fn default() -> Self {
        Self {
            enabled: STAGES.map(str::to_owned).into(),
            exempt_paths: vec![
                "/health".into(),
                "/robots.txt".into(),
                "/sitemap.xml".into(),
            ],
            exempt_ips: HashSet::new(),
            static_keywords: [".php", "xmlrpc", "wp-", ".env", ".git", ".bak", "shell"]
                .map(str::to_owned)
                .into(),
            dynamic_keywords: HashSet::new(),
            legitimate_keywords: [
                "profile",
                "users",
                "account",
                "settings",
                "dashboard",
                "search",
                "admin",
                "login",
                "register",
                "api",
            ]
            .map(str::to_owned)
            .into(),
            exempt_keywords: vec![],
            allowed_path_keywords: vec![],
            learning: true,
            geo_allow: HashSet::new(),
            geo_block: HashSet::new(),
            blocklist: HashMap::new(),
            rate_window: 10.0,
            rate_max: 20,
            rate_flood: 40,
            rate_key_mode: "ip_path".into(),
            min_form_time: 1.0,
            max_page_time: 240.0,
            skip_authenticated: true,
            honeypot_field: String::new(),
            login_prefixes: vec!["/login".into(), "/admin/login".into()],
            learn_context: String::new(),
            learn_scan_patterns: vec![],
            learn_static_keywords: vec![],
            post_only_suffixes: vec![],
            allowed_methods: vec![],
            uuid_prefix: "/user/".into(),
            uuid_window: 60.0,
            uuid_threshold: 5,
            anomaly_window: 60.0,
            required_headers: vec!["HTTP_USER_AGENT".into(), "HTTP_ACCEPT".into()],
            required_headers_by_method: HashMap::new(),
            header_min_score: 3,
            header_max_bytes: 32 * 1024,
            header_max_count: 100,
            header_max_user_agent: 500,
            header_max_accept: 4096,
            blocked_user_agents: vec![],
            header_bytes_without_prefix: false,
            header_policy: "python".into(),
            header_skip_extensions: vec![],
            header_skip_paths: vec![],
            header_wasm_check: false,
            suspicious_user_agents: None,
            legitimate_bots: None,
            sql_mode: "block".into(),
            max_bytes: 65536,
            block_duration: 86400.0,
            max_clients: 100000,
            max_history: 1000,
            max_pending: 10000,
            policy_store: "memory".into(),
            route_schema_mode: "off".into(),
            route_manifest: Value::Null,
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    ip: String,
    path: String,
    method: String,
    headers: HashMap<String, String>,
    query: String,
    body: Value,
    now: f64,
    country: String,
    authenticated: bool,
    exists: bool,
    #[serde(default)]
    parsed_query: Value,
    #[serde(default)]
    disabled: HashSet<String>,
    #[serde(default)]
    payload_exhausted: bool,
    #[serde(default)]
    defer_body: bool,
    /// Rate limits from the request's path rule, replacing the profile's where set.
    #[serde(default)]
    rate: Option<RateLimits>,
    /// Whether the record named by a well-formed path UUID exists, from the adapter's resolver.
    #[serde(default)]
    uuid_exists: Option<bool>,
    /// With policy_store "host": the host's blocklist and exemptions for this request, and its
    /// learned keywords and blocked countries.
    #[serde(default)]
    blocked: Option<bool>,
    #[serde(default)]
    exempt: Option<bool>,
    #[serde(default)]
    dynamic_keywords: Option<Vec<String>>,
    #[serde(default)]
    geo_countries: Option<Vec<String>>,
    /// Requests in this request's rate window, this one included, from a shared counter
    /// such as Redis. Replaces the engine's own per-worker count.
    #[serde(default)]
    rate_count: Option<usize>,
    /// The host's verdict on the User-Agent from the app's own pattern lists, matched in the
    /// host's regex dialect: "legitimate", "suspicious:<pattern>" or "" for neither. Replaces
    /// the lists in the "node" header policy.
    #[serde(default)]
    user_agent_verdict: Option<String>,
}
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RateLimits {
    window: Option<f64>,
    max: Option<usize>,
    flood: Option<usize>,
}
struct Pending {
    request: Request,
    status: u16,
    reason: String,
    apply: HashSet<String>,
    exempt: bool,
}
#[derive(Default)]
struct Client {
    rates: HashMap<String, VecDeque<f64>>,
    honey: Option<f64>,
    uuid: VecDeque<(f64, usize)>,
    history: VecDeque<RecentEntryInput>,
    last: f64,
}
#[derive(Serialize)]
struct Begin {
    context: String,
    status: u16,
    reason: String,
    /// A SQL finding allowed through because sql_mode is "monitor", for the adapter to log.
    #[serde(skip_serializing_if = "Option::is_none")]
    monitor: Option<String>,
    /// The route schema review's log line, for the adapter to log.
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<Value>,
}

/// What a route schema review does to a finding in the profile's mode.
enum Verdict {
    Keep,
    Clear,
    Mismatch,
    Unexpected,
}

/// Reviews a payload finding, by where it came from, against the route's saved inputs.
fn reviewed(
    schema: Option<&aiwaf_core::route_risks::RouteRisks>,
    profile: &Profile,
    method: &str,
    path: &str,
    query: &str,
    body: &Value,
    found: Option<(&str, aiwaf_core::route_risks::Source)>,
) -> (Verdict, Option<Value>) {
    use aiwaf_core::route_risks::Review;
    let Some(schema) = schema else { return (Verdict::Keep, None) };
    let review = schema.review_found_method(method, path, query, &body_bytes(body), profile.max_bytes, found);
    let event = review.as_ref().map(|review| review.event(&profile.route_schema_mode, method));
    let enforce = profile.route_schema_mode == "enforce";
    let verdict = match review {
        Some(Review::Mismatch { .. }) if enforce => Verdict::Mismatch,
        Some(Review::Unexpected { .. }) if enforce => Verdict::Unexpected,
        Some(Review::Relax { .. }) if enforce => Verdict::Clear,
        _ => Verdict::Keep,
    };
    (verdict, event)
}

/// A result with the route schema review's log line, when there is one.
fn with_schema(mut result: Value, schema: Option<Value>) -> Value {
    if let (Some(event), Some(object)) = (schema, result.as_object_mut()) {
        object.insert("schema".into(), event);
    }
    result
}

/// A request body as the payload check reads it: the text sent, or a parsed body as JSON.
fn body_bytes(body: &Value) -> Vec<u8> {
    match body {
        Value::Null => Vec::new(),
        Value::String(text) => text.as_bytes().to_vec(),
        other => other.to_string().into_bytes(),
    }
}
pub struct Engine {
    profile: Profile,
    clients: HashMap<String, Client>,
    pending: HashMap<String, Pending>,
    blocks: HashMap<String, f64>,
    learned: HashSet<String>,
    last_cleanup: f64,
    /// The largest rate window used so far, so overrides with longer windows aren't pruned early.
    rate_window: f64,
    processed: usize,
    predictions: usize,
    pub model: Option<IsolationForest>,
    header_limits: HeaderLimits,
    agent_lists: Option<UserAgentLists>,
    /// The route manifest's saved inputs, when route_schema_mode isn't "off".
    schema: Option<aiwaf_core::route_risks::RouteRisks>,
}

/// Whether a static or learned keyword blocks this path. With exempt keywords or allowed path
/// fragments, as the regular check: an allowed fragment in the path skips keyword blocks, and the
/// first static keyword found, then the first learned one, blocks unless it's exempt.
fn keyword_hit(p: &Profile, path: &str, learned: &[&String]) -> bool {
    if p.exempt_keywords.is_empty() && p.allowed_path_keywords.is_empty() {
        return p
            .static_keywords
            .iter()
            .chain(learned.iter().copied())
            .any(|s| path.contains(s.as_str()));
    }
    let lower = path.to_lowercase();
    if p.allowed_path_keywords.iter().any(|fragment| lower.contains(fragment.as_str())) {
        return false;
    }
    let first_static = p.static_keywords.iter().find(|s| path.contains(s.as_str()));
    let first_learned = learned.iter().copied().find(|s| lower.contains(s.as_str()));
    [first_static, first_learned]
        .into_iter()
        .flatten()
        .any(|keyword| !p.exempt_keywords.contains(keyword))
}

/// The segments Node.js's regular middleware learns from a 404 (anomalyDetector
/// maybeLearnKeyword): none on a path with an allowed fragment; otherwise all of them, when the
/// path looks like a scan and has a segment that isn't a static keyword.
fn node_segments(p: &Profile, path: &str) -> Vec<String> {
    static SEGMENTS: OnceLock<Regex> = OnceLock::new();
    let lower = path.to_lowercase();
    if lower.is_empty()
        || p.exempt_keywords.iter().any(String::is_empty)
        || p.allowed_path_keywords.iter().any(|fragment| lower.contains(fragment.as_str()))
    {
        return Vec::new();
    }
    // JavaScript's \W without the u flag: anything but ASCII letters, digits and _.
    let split = SEGMENTS.get_or_init(|| Regex::new(r"[^A-Za-z0-9_]+").unwrap());
    let segments: Vec<&str> = split.split(&lower).filter(|s| s.len() > 3).collect();
    let scanning = p.learn_scan_patterns.iter().any(|pattern| lower.contains(pattern.as_str()))
        || lower.contains("../")
        || lower.contains("..\\")
        || ["%2e%2e", "%252e", "%c0%ae"].iter().any(|s| lower.contains(s));
    if scanning && segments.iter().any(|s| !p.learn_static_keywords.iter().any(|k| k == s)) {
        segments.into_iter().map(str::to_owned).collect()
    } else {
        Vec::new()
    }
}

fn is_post_only(p: &Profile, path: &str) -> bool {
    let path = path.to_lowercase();
    p.post_only_suffixes.iter().any(|suffix| path.ends_with(suffix.as_str()))
}

fn is_login_path(p: &Profile, path: &str) -> bool {
    let path = path.to_lowercase();
    p.login_prefixes.iter().any(|prefix| path.starts_with(prefix.as_str()))
}

impl Engine {
    /// Node.js's regular header check, then aiwaf-wasm's own check as its regular middleware
    /// runs it: the app's User-Agent cap, the same required headers (aiwaf-wasm's defaults when
    /// none are), and no protocol.
    fn node_headers(&self, r: &Request, required: &[String]) -> Option<String> {
        let p = &self.profile;
        let path = r.path.to_lowercase();
        if p.header_skip_extensions.iter().any(|ext| path.ends_with(ext.as_str()))
            || p.header_skip_paths.iter().any(|prefix| path.starts_with(prefix.as_str()))
        {
            return None;
        }
        if p.header_policy == "node_ua" {
            if !p.header_wasm_check {
                return None;
            }
            let mut headers = r.headers.clone();
            headers.remove("SERVER_PROTOCOL");
            return validate_headers_with_config_limits(
                &headers,
                Some(vec![]),
                Some(0),
                &HeaderLimits::default(),
            );
        }
        let min_score = if required.is_empty() { 0 } else { p.header_min_score };
        validate_headers_node(
            &r.headers,
            required,
            min_score,
            &self.header_limits,
            self.agent_lists.as_ref(),
            r.user_agent_verdict.as_deref(),
        )
        .or_else(|| {
            if !p.header_wasm_check {
                return None;
            }
            let mut headers = r.headers.clone();
            headers.remove("SERVER_PROTOCOL");
            validate_headers_with_config_limits(
                &headers,
                (!required.is_empty()).then(|| required.to_vec()),
                Some(min_score),
                &HeaderLimits {
                    max_user_agent_length: self.header_limits.max_user_agent_length,
                    ..HeaderLimits::default()
                },
            )
        })
    }

    pub fn new(profile: &str) -> Result<Self, String> {
        let mut p: Profile = serde_json::from_str(profile).map_err(|e| e.to_string())?;
        if p.enabled.iter().any(|s| !STAGES.contains(&s.as_str())) {
            return Err("Unknown stage".into());
        }
        if !["", "node"].contains(&p.learn_context.as_str()) {
            return Err("Unsupported learn_context".into());
        }
        if !["python", "node", "node_ua"].contains(&p.header_policy.as_str()) {
            return Err("Unsupported header_policy".into());
        }
        if !["off", "block", "monitor"].contains(&p.sql_mode.as_str())
            || !["memory", "host"].contains(&p.policy_store.as_str())
            || !["enforce", "monitor", "off"].contains(&p.route_schema_mode.as_str())
            || !["ip", "ip_path"].contains(&p.rate_key_mode.as_str())
        {
            return Err("Unsupported native policy".into());
        }
        if !(1..=1048576).contains(&p.max_bytes)
            || p.rate_max == 0
            || p.rate_flood < p.rate_max
            || p.rate_flood > 100000
            || p.max_history == 0
            || p.max_history > 10000
            || p.max_clients == 0
            || p.max_pending == 0
            || p.uuid_threshold == 0
        {
            return Err("Invalid native limits".into());
        }
        if [
            p.rate_window,
            p.max_page_time,
            p.uuid_window,
            p.anomaly_window,
            p.block_duration,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
            || !p.min_form_time.is_finite()
            || p.min_form_time < 0.0
        {
            return Err("Invalid native timing".into());
        }
        if p.blocklist.values().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("Invalid blocklist expiry (0 means permanent)".into());
        }
        p.static_keywords = p
            .static_keywords
            .into_iter()
            .map(|s| s.to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        p.legitimate_keywords = p
            .legitimate_keywords
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect();
        p.exempt_paths = p
            .exempt_paths
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect();
        p.geo_allow = p.geo_allow.into_iter().map(|s| s.to_uppercase()).collect();
        p.geo_block = p.geo_block.into_iter().map(|s| s.to_uppercase()).collect();
        for list in [
            &mut p.exempt_keywords,
            &mut p.allowed_path_keywords,
            &mut p.post_only_suffixes,
            &mut p.header_skip_extensions,
            &mut p.header_skip_paths,
        ] {
            *list = list.iter().map(|s| s.to_lowercase()).collect();
        }
        p.required_headers_by_method = p
            .required_headers_by_method
            .into_iter()
            .map(|(method, headers)| (method.to_uppercase(), headers))
            .collect();
        p.login_prefixes = p
            .login_prefixes
            .into_iter()
            .map(|s| s.to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        let header_limits = HeaderLimits {
            max_header_bytes: p.header_max_bytes,
            max_header_count: p.header_max_count,
            max_user_agent_length: p.header_max_user_agent,
            max_accept_length: p.header_max_accept,
            blocked_user_agents: p
                .blocked_user_agents
                .iter()
                .map(|agent| agent.trim().to_lowercase())
                .filter(|agent| !agent.is_empty())
                .collect(),
            count_http_prefix: !p.header_bytes_without_prefix,
            extra_suspicious: Vec::new(),
        };
        let agent_lists = if p.suspicious_user_agents.is_some() || p.legitimate_bots.is_some() {
            Some(UserAgentLists::compile(
                p.suspicious_user_agents.as_deref(),
                p.legitimate_bots.as_deref(),
            )?)
        } else {
            None
        };
        let blocks = p.blocklist.clone();
        let learned = p
            .dynamic_keywords
            .iter()
            .map(|s| s.to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        let schema = (p.route_schema_mode != "off").then(|| aiwaf_core::route_risks::RouteRisks::new(&p.route_manifest));
        Ok(Self {
            schema,
            profile: p,
            clients: HashMap::new(),
            pending: HashMap::new(),
            blocks,
            learned,
            last_cleanup: 0.0,
            rate_window: 0.0,
            processed: 0,
            predictions: 0,
            model: None,
            header_limits,
            agent_lists,
        })
    }
    /// Reviews a payload finding, by where it came from, against the route's saved inputs.
    fn reviewed(
        &self,
        method: &str,
        path: &str,
        query: &str,
        body: &Value,
        found: Option<(&str, aiwaf_core::route_risks::Source)>,
    ) -> (Verdict, Option<Value>) {
        reviewed(self.schema.as_ref(), &self.profile, method, path, query, body, found)
    }
    pub fn set_model(&mut self, model: IsolationForest) -> Result<(), String> {
        let state = model.to_state();
        if state.n_features_in_ != 6 || state.trees.is_empty() {
            return Err("A fitted six-feature Rust model is required".into());
        }
        self.model = Some(model);
        Ok(())
    }
    fn cleanup(&mut self, now: f64) {
        if now - self.last_cleanup < 30.0 {
            return;
        }
        let rate_window = self.profile.rate_window.max(self.rate_window);
        let keep = rate_window
            .max(self.profile.anomaly_window)
            .max(self.profile.uuid_window)
            .max(self.profile.max_page_time + 300.0);
        self.clients.retain(|_, c| now - c.last <= keep);
        for c in self.clients.values_mut() {
            c.rates.retain(|_, times| {
                times.retain(|t| now - *t < rate_window);
                !times.is_empty()
            });
            c.uuid.retain(|(t, _)| now - *t < self.profile.uuid_window);
            c.history
                .retain(|e| now - e.timestamp < self.profile.anomaly_window);
        }
        self.blocks
            .retain(|_, expires| *expires == 0.0 || *expires > now);
        // Pending count is bounded. Live streams retain their context until finish/close.
        self.last_cleanup = now;
    }
    pub fn inspect_payload(&self, payload: &str) -> Result<String, String> {
        if payload.len()
            > self
                .profile
                .max_bytes
                .saturating_mul(8)
                .saturating_add(8192)
        {
            return Ok(json!({"status":413,"rule":"payload_inspection_limit"}).to_string());
        }
        let value: Value = serde_json::from_str(payload).map_err(|e| e.to_string())?;
        let result = if self.profile.sql_mode == "off" {
            None
        } else {
            inspect(&value, self.profile.max_bytes)
        };
        let monitor = self.profile.sql_mode == "monitor";
        Ok(result
            .map(|(status, rule)| {
                if monitor {
                    json!({"status":status,"rule":rule,"mode":"monitor"})
                } else {
                    json!({"status":status,"rule":rule})
                }
            })
            .unwrap_or(Value::Null)
            .to_string())
    }
    pub fn begin(&mut self, input: &str) -> Result<String, String> {
        if input.len()
            > self
                .profile
                .max_bytes
                .saturating_mul(8)
                .saturating_add(65536)
        {
            return Err("Request envelope limit exceeded".into());
        }
        let mut r: Request = serde_json::from_str(input).map_err(|e| e.to_string())?;
        if !r.now.is_finite() || r.ip.len() > 512 || r.path.len() > 8192 {
            return Err("Invalid request facts".into());
        }
        if r.disabled.iter().any(|s| !STAGES.contains(&s.as_str())) {
            return Err("Unknown disabled stage".into());
        }
        let limits = r.rate.unwrap_or_default();
        let (rate_window, rate_max, rate_flood) = (
            limits.window.unwrap_or(self.profile.rate_window),
            limits.max.unwrap_or(self.profile.rate_max),
            limits.flood.unwrap_or(self.profile.rate_flood),
        );
        if !rate_window.is_finite()
            || rate_window <= 0.0
            || rate_max == 0
            || rate_flood < rate_max
            || rate_flood > 100000
        {
            return Err("Invalid path-rule rate limit".into());
        }
        self.rate_window = self.rate_window.max(rate_window);
        r.path = r.path.to_lowercase();
        r.method = r.method.to_uppercase();
        self.cleanup(r.now);
        if self.pending.len() >= self.profile.max_pending {
            return Err("Pending request limit exceeded".into());
        }
        if self.clients.len() >= self.profile.max_clients && !self.clients.contains_key(&r.ip) {
            return Err("Client state limit exceeded".into());
        }
        let mut apply: HashSet<String> = self
            .profile
            .enabled
            .difference(&r.disabled)
            .cloned()
            .collect();
        let exempt = self.profile.exempt_ips.contains(&r.ip)
            || r.exempt == Some(true)
            || self
                .profile
                .exempt_paths
                .iter()
                .any(|p| r.path.starts_with(p));
        if exempt {
            apply.retain(|s| s == "logging");
        }
        let host = self.profile.policy_store == "host";
        let stored_countries: HashSet<String> = r
            .geo_countries
            .take()
            .unwrap_or_default()
            .into_iter()
            .map(|c| c.trim().to_uppercase())
            .collect();
        let host_keywords: Vec<String> = r
            .dynamic_keywords
            .take()
            .unwrap_or_default()
            .into_iter()
            .map(|k| k.to_lowercase())
            .filter(|k| !k.is_empty())
            .collect();
        let learned: Vec<&String> = if host {
            host_keywords.iter().collect()
        } else {
            self.learned.iter().collect()
        };
        let blocklisted = if host {
            r.blocked == Some(true)
        } else {
            self.blocks
                .get(&r.ip)
                .is_some_and(|expiry| *expiry == 0.0 || *expiry > r.now)
        };
        let mut status = 0u16;
        let mut reason = String::new();
        let mut monitor = None;
        let mut schema = None;
        let p = &self.profile;
        let country = r.country.to_uppercase();
        if apply.contains("geo_block")
            && !country.is_empty()
            && (if p.geo_allow.is_empty() {
                p.geo_block.contains(&country) || stored_countries.contains(&country)
            } else {
                !p.geo_allow.contains(&country)
            })
        {
            status = 403;
            reason = "geo_block".into();
        }
        if status == 0 && apply.contains("ip_keyword_block") {
            if blocklisted {
                status = 403;
                reason = "blocklist".into();
            } else if let Some(k) = validate_url(&r.path).or_else(|| validate_url(&r.query)) {
                status = 403;
                reason = k.into();
            } else if p.sql_mode != "off" {
                use aiwaf_core::route_risks::Source;
                let found: Option<(u16, String, Source)> = if r.payload_exhausted {
                    Some((413, "payload_inspection_limit".into(), Source::Payload))
                } else if let Some((s, k)) =
                    inspect(&json!([r.query, r.body, r.parsed_query]), p.max_bytes)
                {
                    Some((s, k.into(), Source::Payload))
                } else if let Some((k, s)) = aiwaf_core::payload::inspect_sql_headers(
                    r.headers
                        .iter()
                        .map(|(name, value)| (name.as_str(), value.as_str())),
                    p.max_bytes,
                ) {
                    // Cookies, the Referer and other headers, as in Python and Node.js.
                    Some((s as _, k, Source::Headers))
                } else {
                    aiwaf_core::payload::inspect_sql_path(&r.path, p.max_bytes)
                        .map(|(k, s)| (s as _, k, Source::Path))
                };
                // The route's saved inputs review the finding: see AIWAF_ROUTE_SCHEMA_MODE. A
                // deferred body is reviewed when it's inspected.
                let (verdict, event) = if r.payload_exhausted || (r.defer_body && found.is_none()) {
                    (Verdict::Keep, None)
                } else {
                    self.reviewed(
                        &r.method,
                        &r.path,
                        &r.query,
                        &r.body,
                        found.as_ref().filter(|(s, _, _)| *s == 403).map(|(_, k, source)| (k.as_str(), *source)),
                    )
                };
                schema = event;
                let finding = match verdict {
                    Verdict::Keep => found.map(|(s, k, _)| (s, k)),
                    Verdict::Clear => None,
                    Verdict::Mismatch => Some((403, "schema_type_mismatch".into())),
                    Verdict::Unexpected => Some((403, "schema_unexpected_input".into())),
                };
                match finding {
                    Some((_, rule)) if p.sql_mode == "monitor" => monitor = Some(rule),
                    Some((s, rule)) => {
                        status = s;
                        reason = rule;
                    }
                    None => {}
                }
            }
            if status == 0
                && !p
                    .legitimate_keywords
                    .iter()
                    .any(|s| r.path.trim_matches('/') == s)
                && keyword_hit(p, &r.path, &learned)
            {
                status = 403;
                reason = "keyword".into();
            }
        }
        let c = self.clients.entry(r.ip.clone()).or_default();
        c.last = r.now;
        if status == 0 && apply.contains("rate_limit") {
            let key = if p.rate_key_mode == "ip" {
                String::new()
            } else {
                r.path.clone()
            };
            if let Some(burst) = r.rate_count {
                if burst > rate_flood {
                    status = 403;
                    reason = "flood".into();
                } else if burst > rate_max {
                    status = 429;
                    reason = "rate_limit".into();
                }
            // Bound path buckets even for a single hostile client.
            } else if c.rates.len() >= p.max_history && !c.rates.contains_key(&key) {
                status = 429;
                reason = "rate_bucket_limit".into();
            } else {
                let times = c.rates.entry(key).or_default();
                while times.front().is_some_and(|t| r.now - *t >= rate_window) {
                    times.pop_front();
                }
                // A denied flood cannot grow an unbounded timestamp list.
                if times.len() <= rate_flood {
                    times.push_back(r.now);
                }
                let burst = times.len();
                if burst > rate_flood {
                    status = 403;
                    reason = "flood".into();
                } else if burst > rate_max {
                    status = 429;
                    reason = "rate_limit".into();
                }
            }
        }
        if status == 0 && apply.contains("ai_anomaly") && blocklisted {
            status = 403;
            reason = "blocklist".into();
        }
        if status == 0 && apply.contains("honeypot") && !r.defer_body {
            if !p.allowed_methods.is_empty() && !p.allowed_methods.contains(&r.method) {
                status = 405;
                reason = "method_not_allowed".into();
            } else if r.method == "GET" && is_post_only(p, &r.path) {
                status = 405;
                reason = "get_post_only".into();
            } else if !p.honeypot_field.is_empty()
                && r.body
                    .get(&p.honeypot_field)
                    .is_some_and(|v| !v.is_null() && v != false && v != "" && v != 0)
            {
                status = 403;
                reason = "honeypot_field".into();
            } else if !(p.skip_authenticated && r.authenticated) {
                if r.method == "GET" {
                    c.honey = Some(r.now);
                }
                if r.method == "POST" {
                    if let Some(start) = c.honey {
                        let elapsed = (r.now - start).max(0.0);
                        let min =
                            if is_login_path(p, &r.path) {
                                p.min_form_time.min(0.1)
                            } else {
                                p.min_form_time
                            };
                        if elapsed > p.max_page_time {
                            c.honey = None;
                            status = 409;
                            reason = "page_expired".into();
                        } else if elapsed < min {
                            status = 403;
                            reason = "honeypot_too_fast".into();
                        }
                    }
                }
            }
        }
        if status == 0 && apply.contains("uuid_tamper") && r.path.starts_with(&p.uuid_prefix) {
            let uid = r.path[p.uuid_prefix.len()..].trim_end_matches('/');
            static UUID: OnceLock<Regex> = OnceLock::new();
            let valid = UUID.get_or_init(|| {
                Regex::new(r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
                    .unwrap()
            });
            if !uid.contains('/') && !valid.is_match(uid) {
                status = 403;
                reason = "uuid_malformed".into();
            } else if !uid.contains('/') && r.uuid_exists == Some(false) {
                // The record doesn't exist: blocked, as the regular middleware's resolver check does.
                status = 403;
                reason = "uuid_not_found".into();
            }
        }
        if status == 0 && apply.contains("header_validation") {
            let required = p
                .required_headers_by_method
                .get(&r.method.to_uppercase())
                .unwrap_or(&p.required_headers);
            let found = if p.header_policy != "python" {
                self.node_headers(&r, required)
            } else {
                validate_headers_full(&r.headers, required, p.header_min_score, &self.header_limits)
            };
            if let Some(k) = found {
                status = 403;
                reason = format!("header:{k}");
            }
        }
        if status == 403 && reason != "blocklist" {
            self.blocks.insert(r.ip.clone(), r.now + p.block_duration);
        }
        // Only response facts are retained; streaming responses must not pin request payloads.
        r.body = Value::Null;
        r.parsed_query = Value::Null;
        r.query.clear();
        r.headers.retain(|key, _| key == "HTTP_USER_AGENT");
        let context = NEXT_CONTEXT.fetch_add(1, Ordering::Relaxed).to_string();
        self.pending.insert(
            context.clone(),
            Pending {
                request: r,
                status,
                reason: reason.clone(),
                apply,
                exempt,
            },
        );
        self.processed += 1;
        serde_json::to_string(&Begin {
            context,
            status,
            reason,
            monitor,
            schema,
        })
        .map_err(|e| e.to_string())
    }
    pub fn inspect_body(
        &mut self,
        context: &str,
        payload: &str,
        exhausted: bool,
        authenticated: bool,
        now: f64,
    ) -> Result<String, String> {
        if !now.is_finite() {
            return Err("Invalid body time".into());
        }
        let item = self
            .pending
            .get_mut(context)
            .ok_or("Unknown, foreign or completed context")?;
        if !item.request.defer_body {
            return Err("Request body is not deferred".into());
        }
        item.request.defer_body = false;
        if item.status != 0 {
            return Ok(json!({"status":item.status,"rule":item.reason}).to_string());
        }
        let limit = payload.len()
            > self
                .profile
                .max_bytes
                .saturating_mul(8)
                .saturating_add(8192);
        let body: Value = if limit || exhausted {
            Value::Null
        } else {
            serde_json::from_str(payload).map_err(|e| e.to_string())?
        };
        let mut finding = None;
        let mut monitored = None;
        let mut schema = None;
        if item.apply.contains("ip_keyword_block") && self.profile.sql_mode != "off" {
            finding = if limit || exhausted {
                Some((413, "payload_inspection_limit"))
            } else {
                inspect(&body, self.profile.max_bytes)
            };
            if !(limit || exhausted) && self.schema.is_some() {
                // The deferred body, reviewed against the route's saved inputs.
                let r = &item.request;
                let found = finding.filter(|(s, _)| *s == 403).map(|(_, k)| (k, aiwaf_core::route_risks::Source::Payload));
                let (method, path, query) = (r.method.clone(), r.path.clone(), r.query.clone());
                let (verdict, event) = reviewed(self.schema.as_ref(), &self.profile, &method, &path, &query, &body, found);
                schema = event;
                match verdict {
                    Verdict::Keep => {}
                    Verdict::Clear => finding = None,
                    Verdict::Mismatch => finding = Some((403, "schema_type_mismatch")),
                    Verdict::Unexpected => finding = Some((403, "schema_unexpected_input")),
                }
            }
            if self.profile.sql_mode == "monitor" {
                monitored = finding.take();
            }
        }
        if finding.is_none() && item.apply.contains("honeypot") {
            let r = &item.request;
            let p = &self.profile;
            let c = self.clients.entry(r.ip.clone()).or_default();
            if !p.allowed_methods.is_empty() && !p.allowed_methods.contains(&r.method) {
                finding = Some((405, "method_not_allowed"));
            } else if r.method == "GET" && is_post_only(p, &r.path) {
                finding = Some((405, "get_post_only"));
            } else if !p.honeypot_field.is_empty()
                && body
                    .get(&p.honeypot_field)
                    .is_some_and(|v| !v.is_null() && v != false && v != "" && v != 0)
            {
                finding = Some((403, "honeypot_field"));
            } else if !(p.skip_authenticated && authenticated) {
                if r.method == "GET" {
                    c.honey = Some(now);
                }
                if r.method == "POST" {
                    if let Some(start) = c.honey {
                        let min =
                            if is_login_path(p, &r.path) {
                                p.min_form_time.min(0.1)
                            } else {
                                p.min_form_time
                            };
                        if now - start > p.max_page_time {
                            c.honey = None;
                            finding = Some((409, "page_expired"));
                        } else if (now - start).max(0.0) < min {
                            finding = Some((403, "honeypot_too_fast"));
                        }
                    }
                }
            }
        }
        if let Some((status, rule)) = finding {
            item.status = status;
            item.reason = rule.into();
            if status == 403 {
                self.blocks
                    .insert(item.request.ip.clone(), now + self.profile.block_duration);
            }
            return Ok(with_schema(json!({"status":status,"rule":rule}), schema).to_string());
        }
        if let Some((status, rule)) = monitored {
            return Ok(with_schema(json!({"status":status,"rule":rule,"mode":"monitor"}), schema).to_string());
        }
        // A cleared finding or a log line alone: the adapter logs it and lets the request through.
        Ok(schema.map(|event| json!({"schema": event}).to_string()).unwrap_or_else(|| "null".into()))
    }
    pub fn finish(
        &mut self,
        context: &str,
        status: u16,
        elapsed_ms: f64,
        now: f64,
    ) -> Result<String, String> {
        if !(100..=599).contains(&status)
            || !elapsed_ms.is_finite()
            || elapsed_ms < 0.0
            || !now.is_finite()
        {
            return Err("Invalid response facts".into());
        }
        let item = self
            .pending
            .remove(context)
            .ok_or("Unknown, foreign or completed context")?;
        let r = &item.request;
        let host = self.profile.policy_store == "host";
        let c = self.clients.entry(r.ip.clone()).or_default();
        c.last = now;
        let mut block = if item.status == 403 && item.reason != "blocklist" {
            Some((
                item.reason.clone(),
                self.blocks
                    .get(&r.ip)
                    .copied()
                    .unwrap_or(now + self.profile.block_duration),
            ))
        } else {
            None
        };
        if item.status == 0
            && item.apply.contains("uuid_tamper")
            && r.path.starts_with(&self.profile.uuid_prefix)
        {
            while c
                .uuid
                .front()
                .is_some_and(|(t, _)| now - *t >= self.profile.uuid_window)
            {
                c.uuid.pop_front();
            }
            if status == 404 {
                c.uuid.push_back((now, 1));
            } else if status < 400 {
                c.uuid.pop_front();
                c.uuid.pop_front();
            }
            if c.uuid.iter().map(|(_, s)| s).sum::<usize>() >= self.profile.uuid_threshold {
                block = Some((
                    "uuid_repeated_404".into(),
                    now + self.profile.block_duration,
                ));
            }
        }
        let mut learned = None;
        if item.status == 0 && item.apply.contains("ai_anomaly") {
            while c
                .history
                .front()
                .is_some_and(|e| now - e.timestamp >= self.profile.anomaly_window)
            {
                c.history.pop_front();
            }
            if c.history.len() >= self.profile.max_history {
                c.history.pop_front();
            }
            c.history.push_back(RecentEntryInput {
                path_lower: r.path.clone(),
                timestamp: now,
                status: status as i32,
                kw_check: !r.exists,
            });
            let hits = self
                .profile
                .static_keywords
                .iter()
                .filter(|s| r.path.contains(*s))
                .count();
            if let Some(model) = &self.model {
                // Preserve Node's model contract: status before response time (milliseconds).
                let code = [200, 403, 404, 500]
                    .iter()
                    .position(|v| *v == status)
                    .map(|v| v as f64)
                    .unwrap_or(-1.0);
                let burst = c
                    .history
                    .iter()
                    .filter(|e| now - e.timestamp < 10.0)
                    .count();
                let f = vec![
                    r.path.encode_utf16().count() as f64,
                    hits as f64,
                    code,
                    elapsed_ms,
                    burst as f64,
                    c.history.iter().filter(|e| e.status == 404).count() as f64,
                ];
                let anomaly = model.predict(&[f])[0] == -1;
                self.predictions += 1;
                if !r.exists
                    && anomaly
                    && analyze_recent_behavior(
                        c.history.iter().cloned().collect(),
                        self.profile.static_keywords.clone(),
                    )
                    .is_some_and(|s| s.should_block)
                {
                    block = Some(("ai_anomaly".into(), now + self.profile.block_duration));
                }
            }
            if self.profile.learning
                && self.profile.learn_context.is_empty()
                && status == 404
                && !r.exists
                && hits > 0
                && (host || self.learned.len() < 4096)
            {
                // Only learn a probe segment, never arbitrary application vocabulary.
                if let Some(segment) = r.path.split('/').find(|s| {
                    s.len() >= 3 && self.profile.static_keywords.iter().any(|k| s.contains(k))
                }) {
                    if !self
                        .profile
                        .legitimate_keywords
                        .iter()
                        .any(|s| segment.contains(s))
                        && (host || self.learned.insert(segment.to_owned()))
                    {
                        learned = Some(segment.to_owned());
                    }
                }
            }
        }
        let mut node_learned: Vec<String> = Vec::new();
        if self.profile.learn_context == "node"
            && self.profile.learning
            && item.status == 0
            && !item.exempt
            && status == 404
        {
            node_learned = node_segments(&self.profile, &r.path);
            if !host {
                node_learned.retain(|_| self.learned.len() < 4096);
                self.learned.extend(node_learned.iter().cloned());
            }
        }
        if let Some((_, expiry)) = &block {
            self.blocks.insert(r.ip.clone(), *expiry);
        }
        let blocked = item.status != 0;
        let log = if item.apply.contains("logging") {
            Some(csv_row(&[
                now.to_string(),
                r.ip.clone(),
                r.method.clone(),
                r.path.clone(),
                status.to_string(),
                elapsed_ms.to_string(),
                blocked.to_string(),
                item.reason.clone(),
                r.country.clone(),
                r.headers
                    .get("HTTP_USER_AGENT")
                    .cloned()
                    .unwrap_or_default(),
            ]))
        } else {
            None
        };
        let block_csv = block
            .as_ref()
            .map(|(reason, expiry)| csv_row(&[r.ip.clone(), expiry.to_string(), reason.clone()]));
        let keyword_csv = if node_learned.is_empty() {
            learned.as_ref().map(|keyword| csv_row(&[keyword.clone()]))
        } else {
            Some(node_learned.iter().map(|keyword| csv_row(&[keyword.clone()])).collect::<String>())
        };
        // Structured effects for host stores, alongside the CSV rows.
        let block_effect = block
            .as_ref()
            .map(|(reason, expiry)| json!({"ip":r.ip,"reason":reason,"expires_at":expiry}));
        Ok(json!({"log_csv":log,"block_csv":block_csv,"keyword_csv":keyword_csv,"block":block_effect,"keyword":learned,"keywords":node_learned}).to_string())
    }
    pub fn diagnostics(&self) -> String {
        json!({"native":true,"model":self.model.is_some(),"processed_requests":self.processed,"model_predictions":self.predictions,"clients":self.clients.len(),"pending":self.pending.len(),"blocked":self.blocks.len(),"learned_keywords":self.learned.len()}).to_string()
    }
}
fn csv_row(values: &[String]) -> String {
    values
        .iter()
        .map(|v| {
            if v.contains([',', '"', '\n', '\r']) {
                format!("\"{}\"", v.replace('"', "\"\""))
            } else {
                v.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
        + "\n"
}
fn decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h = (bytes[i + 1] as char).to_digit(16);
            let l = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (h, l) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
fn inspect(value: &Value, max_bytes: usize) -> Option<(u16, &'static str)> {
    fn visit(
        v: &Value,
        depth: usize,
        nodes: &mut usize,
        bytes: &mut usize,
        max: usize,
    ) -> Option<(u16, &'static str)> {
        *nodes += 1;
        if *nodes > 4096 || depth > 16 {
            return Some((413, "payload_inspection_limit"));
        }
        match v {
            Value::String(s) => {
                if s.contains('\0') {
                    return Some((403, "content_nul_byte"));
                }
                *bytes += s.len();
                if *bytes > max {
                    return Some((413, "payload_inspection_limit"));
                }
                // A multipart form, part by part: its shape, then each text field as one value.
                if let Some(parts) = aiwaf_core::payload::multipart_parts(s.as_bytes()) {
                    for part in parts {
                        if let Some(rule) = aiwaf_core::payload::part_rule(&part) {
                            return Some((403, rule));
                        }
                        if part.filename.is_none() {
                            let value = String::from_utf8_lossy(&part.content);
                            if let Some(rule) = aiwaf_core::rce::classify(&value).or_else(|| aiwaf_core::sql::classify(&value)) {
                                return Some((403, rule));
                            }
                        }
                    }
                    return None;
                }
                let normalized = decode(&decode(s));
                if normalized.trim_start().starts_with(['{', '[']) {
                    if let Ok(parsed) = serde_json::from_str::<Value>(&normalized) {
                        if let Some(k) = visit(&parsed, depth + 1, nodes, bytes, max) {
                            return Some(k);
                        }
                    }
                }
                // Each form field on its own too, as in data=rO0AB... or __proto__[x]=1.
                if let Some(rule) = aiwaf_core::rce::classify_fields(s).or_else(|| aiwaf_core::rce::classify(&normalized)) {
                    return Some((403, rule));
                }
                if let Some(rule) = aiwaf_core::sql::classify_fields(&normalized) {
                    return Some((403, rule));
                }
            }
            Value::Array(items) => {
                for v in items {
                    if let Some(k) = visit(v, depth + 1, nodes, bytes, max) {
                        return Some(k);
                    }
                }
            }
            Value::Object(items) => {
                // {"__proto__": {...}} or {"constructor": {"prototype": {...}}}.
                if items.contains_key("__proto__")
                    || items.get("constructor").and_then(Value::as_object).is_some_and(|c| c.contains_key("prototype"))
                {
                    return Some((403, "rce_prototype_pollution"));
                }
                for (k, v) in items {
                    if let Some(k) = visit(
                        &Value::String(k.clone()),
                        depth + 1,
                        nodes,
                        bytes,
                        max,
                    )
                    .or_else(|| visit(v, depth + 1, nodes, bytes, max))
                    {
                        return Some(k);
                    }
                }
            }
            _ => {}
        }
        None
    }
    visit(value, 0, &mut 0, &mut 0, max_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(ip: &str, path: &str, method: &str, now: f64, body: Value) -> String {
        json!({"ip":ip,"path":path,"method":method,"now":now,"body":body,"query":"","headers":{"HTTP_USER_AGENT":"Mozilla/5.0 Chrome/130 Safari/537.36","HTTP_ACCEPT":"text/html","HTTP_ACCEPT_LANGUAGE":"en","HTTP_ACCEPT_ENCODING":"gzip","HTTP_CONNECTION":"keep-alive"},"country":"","exists":true,"authenticated":false}).to_string()
    }
    fn begin(e: &mut Engine, r: String) -> Value {
        serde_json::from_str(&e.begin(&r).unwrap()).unwrap()
    }
    fn finish(e: &mut Engine, d: &Value, status: u16, now: f64) -> Value {
        serde_json::from_str(
            &e.finish(d["context"].as_str().unwrap(), status, 1.0, now)
                .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn startup_limits() {
        for p in [
            r#"{"unknown":true}"#,
            r#"{"rate_backend":"redis"}"#,
            r#"{"sql_mode":"loud"}"#,
            r#"{"rate_flood":1}"#,
            r#"{"max_history":0}"#,
        ] {
            assert!(Engine::new(p).is_err());
        }
    }
    #[test]
    fn host_store_facts_drive_policy_and_effects_report_writes() {
        let profile = json!({"enabled":["ip_keyword_block","geo_block","logging"], "exempt_paths":[], "policy_store":"host", "blocklist":{"203.0.113.90": 0.0}});
        let mut e = Engine::new(&profile.to_string()).unwrap();
        let with = |ip: &str, path: &str, facts: Value| {
            let mut r: Value =
                serde_json::from_str(&request(ip, path, "GET", 1.0, Value::Null)).unwrap();
            for (k, v) in facts.as_object().unwrap() {
                r[k] = v.clone();
            }
            r.to_string()
        };
        // In host mode the host's blocked fact replaces the engine's own blocklist.
        assert_eq!(
            begin(&mut e, with("203.0.113.90", "/a", json!({})))["status"],
            0
        );
        assert_eq!(
            begin(&mut e, with("203.0.113.91", "/a", json!({"blocked": true})))["reason"],
            "blocklist"
        );
        assert_eq!(
            begin(
                &mut e,
                with(
                    "203.0.113.92",
                    "/x/zzprobe",
                    json!({"dynamic_keywords":["ZZPROBE"]})
                )
            )["reason"],
            "keyword"
        );
        assert_eq!(
            begin(&mut e, with("203.0.113.93", "/x/zzprobe", json!({})))["status"],
            0
        );
        let mut geo: Value = serde_json::from_str(&with(
            "203.0.113.94",
            "/a",
            json!({"geo_countries":["xx "]}),
        ))
        .unwrap();
        geo["country"] = json!("XX");
        assert_eq!(begin(&mut e, geo.to_string())["reason"], "geo_block");
        // An exempt fact keeps logging only.
        let d = begin(
            &mut e,
            with(
                "203.0.113.91",
                "/a",
                json!({"blocked": true, "exempt": true}),
            ),
        );
        assert_eq!(d["status"], 0);
        // A non-blocklist 403 reports a structured block for the host to write.
        let d = begin(
            &mut e,
            with(
                "203.0.113.95",
                "/x/zzprobe",
                json!({"dynamic_keywords":["zzprobe"]}),
            ),
        );
        let effects = finish(&mut e, &d, 403, 2.0);
        assert_eq!(effects["block"]["ip"], "203.0.113.95");
        assert_eq!(effects["block"]["reason"], "keyword");
        assert!(effects["block"]["expires_at"].as_f64().unwrap() > 2.0);
        assert!(effects["keyword"].is_null());
        assert!(Engine::new(r#"{"policy_store":"redis"}"#).is_err());
    }
    #[test]
    fn a_shared_rate_count_replaces_the_local_count() {
        let profile =
            json!({"enabled":["rate_limit"], "exempt_paths":[], "rate_max": 2, "rate_flood": 4});
        let mut e = Engine::new(&profile.to_string()).unwrap();
        let with = |count: usize| {
            let mut r: Value =
                serde_json::from_str(&request("203.0.113.60", "/a", "GET", 1.0, Value::Null))
                    .unwrap();
            r["rate_count"] = json!(count);
            r.to_string()
        };
        assert_eq!(begin(&mut e, with(2))["status"], 0);
        assert_eq!(begin(&mut e, with(3))["reason"], "rate_limit");
        assert_eq!(begin(&mut e, with(5))["reason"], "flood");
        // Shared counts don't touch the local buckets.
        assert_eq!(begin(&mut e, with(1))["status"], 0);
        for _ in 0..2 {
            assert_eq!(
                begin(
                    &mut e,
                    request("203.0.113.60", "/a", "GET", 1.0, Value::Null)
                )["status"],
                0
            );
        }
    }
    #[test]
    fn a_resolver_miss_blocks_and_a_hit_or_no_resolver_passes() {
        let profile = json!({"enabled":["uuid_tamper"], "exempt_paths":[]});
        let mut e = Engine::new(&profile.to_string()).unwrap();
        let path = "/user/550e8400-e29b-41d4-a716-446655440000";
        let with = |exists: Option<bool>, ip: &str| {
            let mut r: Value =
                serde_json::from_str(&request(ip, path, "GET", 1.0, Value::Null)).unwrap();
            if let Some(exists) = exists {
                r["uuid_exists"] = json!(exists);
            }
            r.to_string()
        };
        assert_eq!(
            begin(&mut e, with(Some(false), "203.0.113.80"))["reason"],
            "uuid_not_found"
        );
        assert_eq!(begin(&mut e, with(Some(true), "203.0.113.81"))["status"], 0);
        assert_eq!(begin(&mut e, with(None, "203.0.113.82"))["status"], 0);
    }
    #[test]
    fn monitor_mode_reports_sql_findings_without_blocking() {
        let profile =
            json!({"enabled":["ip_keyword_block"], "exempt_paths":[], "sql_mode":"monitor"});
        let mut e = Engine::new(&profile.to_string()).unwrap();
        let attack = json!({"q": "' OR 1=1--"});
        let d = begin(
            &mut e,
            request("203.0.113.70", "/api", "POST", 1.0, attack.clone()),
        );
        assert_eq!(d["status"], 0);
        assert_eq!(d["monitor"], "sql_breakout_single_quote");
        // A clean request carries no monitor field, and nothing was blocklisted.
        let d = begin(
            &mut e,
            request("203.0.113.70", "/api", "GET", 2.0, Value::Null),
        );
        assert!(d.get("monitor").is_none());
        assert_eq!(d["status"], 0);
        let payload =
            serde_json::from_str::<Value>(&e.inspect_payload(&attack.to_string()).unwrap())
                .unwrap();
        assert_eq!(payload["mode"], "monitor");
    }
    #[test]
    fn blocklist_and_previous_wire_key_enforce_identically() {
        for key in ["blocklist", "blacklist"] {
            let mut profile = json!({"enabled":["ip_keyword_block"], "exempt_paths":[]});
            profile[key] = json!({"8.8.8.8": 1000.0});
            let mut engine = Engine::new(&profile.to_string()).unwrap();
            let decision = begin(&mut engine, request("8.8.8.8", "/api", "GET", 100.0, json!(null)));
            assert_eq!(decision["status"], 403);
        }
    }
    #[test]
    fn sql_in_checked_headers_is_blocked() {
        let profile = json!({"enabled":["ip_keyword_block"], "exempt_paths":[]}).to_string();
        let mut engine = Engine::new(&profile).unwrap();
        let mut r: Value = serde_json::from_str(&request("8.8.4.4", "/api", "GET", 100.0, json!(null))).unwrap();
        r["headers"]["HTTP_COOKIE"] = json!("theme=dark; session=' OR '1'='x'--");
        let decision = begin(&mut engine, r.to_string());
        assert_eq!(decision["status"], 403);
        let mut ok: Value = serde_json::from_str(&request("8.8.4.5", "/api", "GET", 100.0, json!(null))).unwrap();
        ok["headers"]["HTTP_COOKIE"] = json!("_ga=GA1.2.1.1; csrftoken=abc123");
        ok["headers"]["HTTP_AUTHORIZATION"] = json!("Bearer ' OR 1=1--");
        let decision = begin(&mut engine, ok.to_string());
        assert_ne!(decision["status"], 403);
        let mut other: Value = serde_json::from_str(&request("8.8.4.6", "/api", "GET", 100.0, json!(null))).unwrap();
        other["headers"]["HTTP_X_FORWARDED_FOR"] = json!("127.0.0.1' AND SLEEP(5)--");
        assert_eq!(begin(&mut engine, other.to_string())["status"], 403);
    }

    #[test]
    fn sql_in_the_path_is_blocked() {
        let profile = json!({"enabled":["ip_keyword_block"], "exempt_paths":[]}).to_string();
        let mut engine = Engine::new(&profile).unwrap();
        let r = request("8.8.4.7", "/post/1234%20OR%201=1", "GET", 100.0, json!(null));
        assert_eq!(begin(&mut engine, r)["status"], 403);
        let ok = request("8.8.4.8", "/users/o'reilly", "GET", 100.0, json!(null));
        assert_ne!(begin(&mut engine, ok)["status"], 403);
    }
    #[test]
    fn structured_sql_and_limits() {
        assert_eq!(
            inspect(&json!({"a":{"b":"%2527+OR+1%253D1--"}}), 65536)
                .unwrap()
                .0,
            403
        );
        assert!(inspect(&json!({"a":"%€ ordinary UTF-8"}), 65536).is_none());
        assert_eq!(inspect(&json!({"a":"abcdef"}), 3).unwrap().0, 413);
        assert_eq!(
            inspect(&json!(r#"{"x":"\u0027 OR 1=1--"}"#), 65536)
                .unwrap()
                .0,
            403
        );
        assert!(inspect(&json!({"name":"O'Reilly","notes":"select a color"}), 65536).is_none());
        assert_eq!(inspect(&json!({"a":{"__proto__":{"x":1}}}), 65536).unwrap().1, "rce_prototype_pollution");
        assert_eq!(inspect(&json!(["__proto__[shell]=/bin/sh"]), 65536).unwrap().1, "rce_prototype_pollution");
        assert_eq!(inspect(&json!(["user=bob&data=rO0ABXNyABFqYXZhLnV0aWwuSGFzaE1hcA"]), 65536).unwrap().1, "rce_serialized_object");
        assert!(inspect(&json!({"constructor":"Acme","prototype":true}), 65536).is_none());
        let form = "--b\r\nContent-Disposition: form-data; name=\"f\"; filename=\"shell.php\"\r\n\r\nx\r\n--b--";
        assert_eq!(inspect(&json!([form]), 65536).unwrap().1, "rce_script_upload");
        let form = "--b\r\nContent-Disposition: form-data; name=\"q\"\r\n\r\n1 OR 1=1\r\n--b--";
        assert_eq!(inspect(&json!([form]), 65536).unwrap().1, "sql_breakout_numeric");
        assert!(inspect(&json!(["--b\r\nContent-Disposition: form-data; name=\"q\"\r\n\r\nhello\r\n--b--"]), 65536).is_none());
        assert_eq!(
            inspect(&json!("embedded\0null"), 65536).unwrap().1,
            "content_nul_byte"
        );
    }
    #[test]
    fn rate_atomic_and_isolation() {
        let mut e = Engine::new(r#"{"rate_max":2,"rate_flood":3}"#).unwrap();
        for (i, want) in [0, 0, 429, 403].into_iter().enumerate() {
            let d = begin(
                &mut e,
                request("a", "/", "GET", 100.0 + i as f64, json!(null)),
            );
            assert_eq!(d["status"], want);
            finish(&mut e, &d, if want == 0 { 200 } else { want }, 104.0);
        }
        let mut other = Engine::new("{}").unwrap();
        assert_eq!(
            begin(&mut other, request("a", "/", "GET", 105.0, json!(null)))["status"],
            0
        );
        assert_eq!(
            begin(&mut e, request("a", "/", "GET", 100000.0, json!(null)))["status"],
            0
        );
    }
    #[test]
    fn honeypot_and_response_context() {
        let mut e = Engine::new("{}").unwrap();
        let mut other = Engine::new("{}").unwrap();
        let d = begin(&mut e, request("a", "/form", "GET", 100.0, json!(null)));
        assert!(
            other
                .finish(d["context"].as_str().unwrap(), 200, 1.0, 101.0)
                .is_err()
        );
        assert!(
            e.finish(d["context"].as_str().unwrap(), 0, 1.0, 101.0)
                .is_err()
        );
        let row = finish(&mut e, &d, 200, 99.0);
        assert!(row["log_csv"].as_str().unwrap().contains(",200,1,false,"));
        assert!(
            e.finish(d["context"].as_str().unwrap(), 200, 1.0, 101.0)
                .is_err()
        );
        let d = begin(&mut e, request("a", "/form", "POST", 100.1, json!({})));
        assert_eq!(d["status"], 403);
        let d = begin(&mut other, request("a", "/form", "POST", 100.1, json!({})));
        assert_eq!(d["status"], 0);
    }
    #[test]
    fn header_limits_blocked_agents_and_login_prefixes_come_from_the_profile() {
        let header = |e: &mut Engine, ip: &str, ua: String| {
            let mut r: Value =
                serde_json::from_str(&request(ip, "/docs", "GET", 100.0, json!(null))).unwrap();
            r["headers"]["HTTP_USER_AGENT"] = json!(ua);
            begin(e, r.to_string())
        };
        let long = format!("Mozilla/5.0 Chrome/130 {}", "a".repeat(600));
        let mut default = Engine::new("{}").unwrap();
        assert_eq!(header(&mut default, "a", long.clone())["status"], 403);
        let mut wide = Engine::new(
            r#"{"header_max_user_agent":1000,"blocked_user_agents":["ExampleBrowser"]}"#,
        )
        .unwrap();
        assert_eq!(header(&mut wide, "a", long)["status"], 0);
        let d = header(&mut wide, "b", "Mozilla/5.0 examplebrowser/2".into());
        assert_eq!(d["status"], 403);
        assert!(d["reason"].as_str().unwrap().contains("Blocked: examplebrowser"));
        let mut narrow = Engine::new(r#"{"header_max_count":2}"#).unwrap();
        assert_eq!(
            header(&mut narrow, "a", "Mozilla/5.0 Chrome/130 Safari".into())["reason"],
            "header:Header count exceeds 2"
        );
        // A form posted 0.5 s after it was fetched is too fast, except on a login form.
        let post = |prefixes: &str, path: &str| {
            let mut e = Engine::new(&format!(r#"{{"login_prefixes":{prefixes}}}"#)).unwrap();
            begin(&mut e, request("a", path, "GET", 100.0, json!(null)));
            begin(&mut e, request("a", path, "POST", 100.5, json!({})))["status"].clone()
        };
        assert_eq!(post(r#"["/login","/admin/login"]"#, "/login/"), 0);
        assert_eq!(post(r#"["/login","/admin/login"]"#, "/signin/"), 403);
        assert_eq!(post(r#"["/SignIn/"]"#, "/signin/"), 0);
        assert_eq!(post(r#"["/signin/"]"#, "/SIGNIN/next"), 0);
        assert_eq!(post(r#"["/signin/"]"#, "/login/"), 403);
        // Header bytes counted as Node.js sends the names: 27 with "HTTP_", 17 without.
        let bytes = |profile: &str| {
            let mut e = Engine::new(profile).unwrap();
            let mut r: Value =
                serde_json::from_str(&request("a", "/docs", "GET", 100.0, json!(null))).unwrap();
            r["headers"] = json!({"HTTP_USER_AGENT":"Mozilla/5.0 Chrome/130 Safari/537.36","HTTP_ACCEPT":"text/html","HTTP_ACCEPT_LANGUAGE":"en","HTTP_ACCEPT_ENCODING":"gzip","HTTP_CONNECTION":"keep-alive"});
            begin(&mut e, r.to_string())["status"].clone()
        };
        assert_eq!(bytes(r#"{"header_max_bytes":135}"#), 403);
        assert_eq!(bytes(r#"{"header_max_bytes":135,"header_bytes_without_prefix":true}"#), 0);
    }
    #[test]
    fn keyword_exemptions_post_only_suffixes_and_method_headers() {
        let status = |profile: &str, path: &str, method: &str| {
            let mut e = Engine::new(profile).unwrap();
            begin(&mut e, request("a", path, method, 100.0, json!(null)))["status"].clone()
        };
        let keywords = r#"{"enabled":["ip_keyword_block"],"static_keywords":[".php","shell"]}"#;
        assert_eq!(status(keywords, "/x.php", "GET"), 403);
        let exempt = r#"{"enabled":["ip_keyword_block"],"static_keywords":[".php","shell"],"exempt_keywords":[".PHP"]}"#;
        assert_eq!(status(exempt, "/x.php", "GET"), 0);
        // The first keyword in list order decides, as in the regular check.
        assert_eq!(status(exempt, "/shell/x.php", "GET"), 0);
        let exempt_second = r#"{"enabled":["ip_keyword_block"],"static_keywords":[".php","shell"],"exempt_keywords":["shell"]}"#;
        assert_eq!(status(exempt_second, "/shell/x.php", "GET"), 403);
        assert_eq!(status(exempt_second, "/shell/x", "GET"), 0);
        let allowed = r#"{"enabled":["ip_keyword_block"],"static_keywords":[".php"],"allowed_path_keywords":["Legacy"]}"#;
        assert_eq!(status(allowed, "/legacy/x.php", "GET"), 0);
        assert_eq!(status(allowed, "/x.php", "GET"), 403);
        let post_only = r#"{"enabled":["honeypot"],"post_only_suffixes":["/Submit/"]}"#;
        assert_eq!(status(post_only, "/form/submit/", "GET"), 405);
        assert_eq!(status(post_only, "/form/submit/", "POST"), 0);
        assert_eq!(status(post_only, "/form/", "GET"), 0);
        let by_method = r#"{"enabled":["header_validation"],"required_headers":["HTTP_X_TOKEN"],"required_headers_by_method":{"get":[]}}"#;
        assert_eq!(status(by_method, "/docs", "GET"), 0);
        assert_eq!(status(by_method, "/docs", "POST"), 403);
        // The app's own User-Agent lists, in the Node.js header check.
        let agent = |profile: &str, ua: &str| {
            let mut e = Engine::new(profile).unwrap();
            let mut r: Value =
                serde_json::from_str(&request("a", "/docs", "GET", 100.0, json!(null))).unwrap();
            r["headers"]["HTTP_USER_AGENT"] = json!(ua);
            begin(&mut e, r.to_string())["status"].clone()
        };
        let node = r#"{"enabled":["header_validation"],"header_policy":"node","required_headers":[]}"#;
        assert_eq!(agent(node, "curl/8.0 Mozilla"), 403);
        let lists = r#"{"enabled":["header_validation"],"header_policy":"node","required_headers":[],"suspicious_user_agents":["^evil"],"legitimate_bots":["MyMonitor"]}"#;
        assert_eq!(agent(lists, "curl/8.0 Mozilla"), 0);
        assert_eq!(agent(lists, "Evil/1.0 Mozilla"), 403);
        assert_eq!(agent(lists, "mymonitor/1 evil"), 0);
        assert!(Engine::new(r#"{"suspicious_user_agents":["(?<=x)y"]}"#).is_err());
        // The host's verdict replaces the lists: curl passes when the host found nothing.
        let verdict = |v: &str, ua: &str| {
            let mut e = Engine::new(node).unwrap();
            let mut r: Value =
                serde_json::from_str(&request("a", "/docs", "GET", 100.0, json!(null))).unwrap();
            r["headers"]["HTTP_USER_AGENT"] = json!(ua);
            r["user_agent_verdict"] = json!(v);
            begin(&mut e, r.to_string())["status"].clone()
        };
        assert_eq!(verdict("", "curl/8.0 Mozilla"), 0);
        assert_eq!(verdict("suspicious:^evil", "Mozilla/5.0 Chrome/130"), 403);
        assert_eq!(verdict("legitimate", "x"), 0);
    }
    #[test]
    fn exemptions_caps_geo_uuid_and_logging() {
        let mut e = Engine::new(r#"{"geo_block":["XX"]}"#).unwrap();
        let mut r: Value =
            serde_json::from_str(&request("a", "/", "GET", 100.0, json!(null))).unwrap();
        r["country"] = json!("XX");
        let d = begin(&mut e, r.to_string());
        assert_eq!(d["status"], 403);
        r["path"] = json!("/health");
        assert_eq!(begin(&mut e, r.to_string())["status"], 0);
        r["ip"] = json!("b");
        r["country"] = json!("");
        r["path"] = json!("/user/not-a-uuid");
        assert_eq!(begin(&mut e, r.to_string())["status"], 403);
        r["ip"] = json!("c");
        r["path"] = json!("/docs");
        r["headers"] = json!({"HTTP_USER_AGENT":"x".repeat(501)});
        assert_eq!(begin(&mut e, r.to_string())["status"], 403);
        assert_eq!(
            csv_row(&["a,b".into(), "a\"b\nc".into()]),
            "\"a,b\",\"a\"\"b\nc\"\n"
        );
    }
    #[test]
    fn deferred_body_and_context_policy() {
        let mut e = Engine::new(r#"{"honeypot_field":"website"}"#).unwrap();
        let mut r: Value =
            serde_json::from_str(&request("a", "/api", "POST", 100.0, json!(null))).unwrap();
        r["defer_body"] = json!(true);
        let d = begin(&mut e, r.to_string());
        assert_eq!(d["status"], 0);
        let context = d["context"].as_str().unwrap();
        let finding: Value = serde_json::from_str(
            &e.inspect_body(
                context,
                r#"{"nested":{"q":"' OR 1=1--"}}"#,
                false,
                false,
                100.0,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(finding["status"], 403);
        assert!(e.inspect_body(context, "{}", false, false, 100.0).is_err());
        let effects = finish(&mut e, &d, 403, 100.0);
        assert!(
            effects["log_csv"]
                .as_str()
                .unwrap()
                .contains(",true,sql_breakout_single_quote,")
        );
        r["ip"] = json!("b");
        r["path"] = json!("/health");
        let d = begin(&mut e, r.to_string());
        assert_eq!(
            e.inspect_body(
                d["context"].as_str().unwrap(),
                r#"{"website":"bot","q":"' OR 1=1--"}"#,
                false,
                false,
                100.0
            )
            .unwrap(),
            "null"
        );
        r["ip"] = json!("c");
        r["path"] = json!("/api");
        let d = begin(&mut e, r.to_string());
        let finding: Value = serde_json::from_str(
            &e.inspect_body(
                d["context"].as_str().unwrap(),
                r#"{"website":"bot"}"#,
                false,
                false,
                100.0,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(finding["rule"], "honeypot_field");
    }
    #[test]
    fn long_stream_survives_state_cleanup() {
        let mut e = Engine::new("{}").unwrap();
        let stream = begin(
            &mut e,
            request("stream", "/stream", "GET", 100.0, json!(null)),
        );
        let other = begin(&mut e, request("other", "/api", "GET", 1000.0, json!(null)));
        finish(&mut e, &other, 200, 1000.0);
        assert!(
            finish(&mut e, &stream, 200, 1001.0)["log_csv"]
                .as_str()
                .unwrap()
                .contains("/stream")
        );
    }
}
