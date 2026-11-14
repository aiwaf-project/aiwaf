//! Rust-owned enforcement with CSV state and optional delegated honeypot cache I/O.
use crate::{csv_runtime as csv, runtime_policy as policy};
use aiwaf_core::payload::decode;
use aiwaf_core::{
    IsolationForest, RecentEntryInput, analyze_recent_behavior, validate_headers_full_default,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

const STAGES: [&str; 8] = [
    "geo_block",
    "ip_keyword_block",
    "rate_limit",
    "ai_anomaly",
    "honeypot",
    "uuid_tamper",
    "header_validation",
    "logging",
];
const BLOCKLIST_HEADERS: [&str; 12] = [
    "ip",
    "reason",
    "reputation_reason",
    "added_date",
    "extended_request_info",
    "score",
    "offenses",
    "blocked_at",
    "expires_at",
    "duration",
    "permanent",
    "reasons",
];
const LOG_HEADERS: [&str; 11] = [
    "timestamp",
    "ip",
    "method",
    "path",
    "status_code",
    "content_length",
    "response_time_ms",
    "referer",
    "user_agent",
    "blocked",
    "block_reason",
];

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Profile {
    data_dir: String,
    enabled: HashSet<String>,
    exempt_paths: Vec<String>,
    exempt_ips: HashSet<String>,
    sql_mode: String,
    max_bytes: usize,
    learning: bool,
    top_n: usize,
    static_keywords: HashSet<String>,
    legitimate_keywords: HashSet<String>,
    exempt_keywords: HashSet<String>,
    safe_prefixes: Vec<String>,
    geo_allow: HashSet<String>,
    geo_block: HashSet<String>,
    rate_backend: String,
    rate_key_mode: String,
    rate_window: f64,
    rate_max: i64,
    rate_flood: i64,
    #[serde(alias = "soft_block_blacklist")]
    soft_block_blocklist: bool,
    ip_blocking: bool,
    block_duration: Option<i64>,
    min_form_time: f64,
    max_page_time: f64,
    skip_authenticated: bool,
    honeypot_backend: String,
    uuid_score_enabled: bool,
    uuid_window: f64,
    uuid_threshold: i64,
    uuid_malformed_weight: i64,
    uuid_not_found_weight: i64,
    uuid_success_decay: i64,
    required_headers: Vec<String>,
    header_min_score: i32,
    anomaly_window: f64,
    anomaly_keywords: Vec<String>,
    log_path: String,
    log_fsync: bool,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            data_dir: "aiwaf_data".into(),
            enabled: STAGES.map(str::to_owned).into(),
            exempt_paths: vec![
                "/health".into(),
                "/robots.txt".into(),
                "/sitemap.xml".into(),
            ],
            exempt_ips: HashSet::new(),
            sql_mode: "block".into(),
            max_bytes: 65536,
            learning: true,
            top_n: 10,
            static_keywords: [
                ".php",
                "xmlrpc",
                "wp-",
                ".env",
                ".git",
                ".bak",
                "shell",
                "filemanager",
            ]
            .map(str::to_owned)
            .into(),
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
            exempt_keywords: HashSet::new(),
            safe_prefixes: vec![],
            geo_allow: HashSet::new(),
            geo_block: HashSet::new(),
            rate_backend: "memory".into(),
            rate_key_mode: "ip_path".into(),
            rate_window: 10.0,
            rate_max: 20,
            rate_flood: 40,
            soft_block_blocklist: false,
            ip_blocking: true,
            block_duration: None,
            min_form_time: 1.0,
            max_page_time: 240.0,
            skip_authenticated: true,
            honeypot_backend: "csv".into(),
            uuid_score_enabled: true,
            uuid_window: 60.0,
            uuid_threshold: 5,
            uuid_malformed_weight: 5,
            uuid_not_found_weight: 1,
            uuid_success_decay: 2,
            required_headers: vec!["HTTP_USER_AGENT".into(), "HTTP_ACCEPT".into()],
            header_min_score: 3,
            anomaly_window: 60.0,
            anomaly_keywords: [
                "admin",
                "wp-admin",
                "wp-content",
                "wp-includes",
                "wp-config",
                "xmlrpc",
                "phpmyadmin",
                "adminer",
                "config",
                "configuration",
                "settings",
                "setup",
                "install",
                "installer",
                "backup",
                "database",
                "mysql",
                "sql",
                "dump",
                ".env",
                ".git",
                ".htaccess",
                ".htpasswd",
                "passwd",
                "shadow",
                "robots",
                "cgi-bin",
                "scripts",
                "shell",
                "cmd",
                "exec",
                "system",
                "eval",
                "base64",
                "union",
                "select",
                "drop",
                "delete",
                "insert",
                "update",
                "script",
                "javascript",
                "onload",
                "onerror",
                "onclick",
                "document",
                "cookie",
                "alert",
            ]
            .map(str::to_owned)
            .to_vec(),
            log_path: String::new(),
            log_fsync: false,
        }
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| (*s).into()).collect()
}
fn err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
#[derive(Clone)]
struct Facts {
    ip: String,
    path: String,
    method: String,
    headers: HashMap<String, String>,
    exists: bool,
    now: f64,
    date: String,
    active: HashSet<String>,
    uuid: Option<String>,
    exempt: bool,
    query: String,
}
#[pyclass]
pub struct NativePipelineRequest {
    owner: u64,
    facts: Facts,
    status: i32,
    reason: String,
    finished: AtomicBool,
    started: std::time::Instant,
}
#[derive(Default)]
struct State {
    rates: HashMap<String, Vec<f64>>,
    honeypot: HashMap<String, (f64, f64)>,
    uuid: HashMap<String, Vec<(f64, i64)>>,
    histories: HashMap<String, Vec<(RecentEntryInput, f64)>>,
    cleanup_at: f64,
    model: Option<Arc<IsolationForest>>,
}
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
#[pyclass]
pub struct NativeCSVPipeline {
    profile: Profile,
    state: Mutex<State>,
    id: u64,
    processed: AtomicU64,
    honeypot_cache: Option<PyObject>,
}

impl NativeCSVPipeline {
    fn cached_honeypot_time(&self, py: Python<'_>, ip: &str, now: f64) -> PyResult<Option<f64>> {
        let cache = self
            .honeypot_cache
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Honeypot cache missing"))?;
        let value: Option<(f64, f64)> = cache
            .call_method1(py, "get_timestamp", (ip,))?
            .extract(py)?;
        match value {
            Some((time, expiry)) if !time.is_finite() || !expiry.is_finite() || expiry < time => {
                Err(PyValueError::new_err("Invalid cached honeypot timestamp"))
            }
            Some((time, expiry)) if now <= expiry => Ok(Some(time)),
            _ => Ok(None),
        }
    }
    fn file(&self, name: &str) -> String {
        Path::new(&self.profile.data_dir)
            .join(name)
            .to_string_lossy()
            .into_owned()
    }
    fn blocklist(&self, py: Python<'_>, f: &Facts, reason: &str) -> PyResult<()> {
        if self.profile.ip_blocking && !f.exempt {
            csv::csv_blocklist_block(
                py,
                self.file("blacklist.csv"),
                strings(&BLOCKLIST_HEADERS),
                f.ip.clone(),
                reason.into(),
                f.date.clone(),
                None,
                f.now,
                self.profile.block_duration,
            )?;
        }
        Ok(())
    }
    fn exempt(&self, py: Python<'_>, ip: &str, path: &str) -> PyResult<bool> {
        if self.profile.exempt_ips.contains(ip)
            || csv::csv_contains(py, self.file("whitelist.csv"), "ip".into(), ip.into(), true)?
        {
            return Ok(true);
        }
        let paths = csv::csv_read_set(py, self.file("path_exemptions.csv"), "path".into(), true)?;
        Ok(self
            .profile
            .exempt_paths
            .iter()
            .chain(paths.iter())
            .any(|p| {
                let p = p.to_lowercase();
                let p = p.trim_end_matches('/');
                path == p || (!p.is_empty() && path.starts_with(&format!("{p}/")))
            }))
    }
    fn uuid_score(&self, py: Python<'_>, ip: &str, now: f64, delta: i64) -> PyResult<i64> {
        if !self.profile.uuid_score_enabled || delta == 0 {
            return Ok(0);
        }
        py.allow_threads(|| {
            let mut state = self.state.lock().map_err(err)?;
            let events = state.uuid.entry(ip.into()).or_default();
            events.retain(|(t, _)| *t >= now - self.profile.uuid_window);
            events.push((now, delta));
            Ok(events.iter().map(|(_, d)| d).sum())
        })
    }
    fn evaluate(
        &self,
        py: Python<'_>,
        f: &Facts,
        query: Vec<u8>,
        body: Vec<u8>,
        exhausted: bool,
        accepts_get: bool,
        accepts_post: bool,
        accepts_method: bool,
        authenticated: bool,
    ) -> PyResult<(i32, Value, String)> {
        let p = &self.profile;
        let block = |reason: String, status: i32, message: Option<String>| -> PyResult<_> {
            self.blocklist(py, f, &reason)?;
            let payload = message
                .map(|m| json!({"error":"blocked", "message":m}))
                .unwrap_or(json!({"error":"blocked"}));
            Ok((status, payload, reason))
        };
        if f.active.contains("geo_block") {
            let country = f.headers.get("AIWAF_COUNTRY").cloned().unwrap_or_default();
            let decision = csv::csv_geo_policy(
                py,
                self.file("geo_blocked_countries.csv"),
                country,
                p.geo_allow.clone(),
                p.geo_block.clone(),
            )?;
            if decision.0 {
                return block(decision.2, 403, None);
            }
        }
        if f.active.contains("ip_keyword_block") {
            if p.ip_blocking
                && csv::csv_blocklist_check(
                    py,
                    self.file("blacklist.csv"),
                    strings(&BLOCKLIST_HEADERS),
                    f.ip.clone(),
                    f.now,
                )?
            {
                return Ok((
                    403,
                    json!({"error":"blocked"}),
                    format!("IP blocklisted: {}", f.ip),
                ));
            }
            if p.sql_mode != "off" {
                if let Some((rule, status)) = py.allow_threads(|| {
                    aiwaf_core::payload::inspect_sql(&query, &body, p.max_bytes, exhausted).or_else(|| {
                        aiwaf_core::payload::inspect_sql_headers(
                            f.headers.iter().map(|(name, value)| (name.as_str(), value.as_str())),
                            p.max_bytes,
                        )
                        .or_else(|| aiwaf_core::payload::inspect_sql_path(&f.path, p.max_bytes))
                    })
                }) {
                    return Ok((status, json!({"error":rule}), rule));
                }
            }
            let mut dynamic = if p.learning {
                csv::csv_read_set(py, self.file("keywords.csv"), "keyword".into(), true)?
                    .into_iter()
                    .collect::<Vec<_>>()
            } else {
                vec![]
            };
            // A stable subset replaces the old process-dependent Python set order.
            dynamic.sort();
            dynamic.truncate(p.top_n);
            let raw = String::from_utf8_lossy(&query).into_owned();
            let keys = query
                .split(|b| *b == b'&')
                .map(|part| {
                    decode(part.split(|b| *b == b'=').next().unwrap_or_default()).to_lowercase()
                })
                .collect();
            let decision = policy::policy_keyword_flask(
                py,
                f.path.clone(),
                raw,
                keys,
                f.exists,
                p.learning,
                p.static_keywords.clone(),
                dynamic.into_iter().collect(),
                p.legitimate_keywords.clone(),
                p.exempt_keywords.clone(),
                p.safe_prefixes.clone(),
                p.static_keywords.clone(),
            );
            if p.learning && !decision.1.is_empty() {
                csv::csv_keywords_update(
                    py,
                    self.file("keywords.csv"),
                    f.date.clone(),
                    decision.1,
                    HashSet::new(),
                )?;
            }
            if let Some(reason) = decision.0 {
                return block(reason, 403, None);
            }
        }
        if f.active.contains("rate_limit") {
            let key = if p.rate_key_mode == "ip" {
                format!("ratelimit:{}", f.ip)
            } else {
                format!("ratelimit:{}:{}", f.ip, f.path)
            };
            let action = if p.rate_backend == "csv" {
                match csv::csv_rate_consume(
                    py,
                    self.file("rate_buckets.csv"),
                    key,
                    f.now,
                    p.rate_window,
                    p.rate_max,
                    p.rate_flood,
                ) {
                    Ok(d) => d.0,
                    Err(_) => {
                        return Ok((
                            503,
                            json!({"error":"temporarily_unavailable"}),
                            "Rate cache unavailable".into(),
                        ));
                    }
                }
            } else {
                py.allow_threads(|| -> PyResult<_> {
                    let mut state = self.state.lock().map_err(err)?;
                    let old = state.rates.remove(&key).unwrap_or_default();
                    let d =
                        policy::policy_rate(old, f.now, p.rate_window, p.rate_max, p.rate_flood);
                    state.rates.insert(key, d.2);
                    Ok(d.0)
                })?
            };
            if action == "flood_block" {
                return block("Flood pattern".into(), 403, None);
            }
            if action == "throttle" {
                if p.soft_block_blocklist {
                    self.blocklist(py, f, "Rate limit exceeded")?;
                }
                return Ok((
                    429,
                    json!({"error":"too_many_requests"}),
                    "Rate limit exceeded".into(),
                ));
            }
        }
        if f.active.contains("ai_anomaly")
            && p.ip_blocking
            && csv::csv_blocklist_check(
                py,
                self.file("blacklist.csv"),
                strings(&BLOCKLIST_HEADERS),
                f.ip.clone(),
                f.now,
            )?
        {
            return Ok((403, json!({"error":"blocked"}), "IP blocklisted".into()));
        }
        if f.active.contains("honeypot") && !(p.skip_authenticated && authenticated) {
            let method_reason = if f.method == "GET"
                && !accepts_get
                && ["/create/", "/submit/", "/upload/", "/delete/", "/process/"]
                    .iter()
                    .any(|s| f.path.ends_with(s))
            {
                Some(format!("GET to obvious POST-only endpoint: {}", f.path))
            } else if f.method == "POST" && !accepts_post {
                Some(format!("POST to GET-only view: {}", f.path))
            } else if !["GET", "POST", "HEAD", "OPTIONS"].contains(&f.method.as_str())
                && !accepts_method
            {
                Some(format!(
                    "{} to view that doesn't support it: {}",
                    f.method, f.path
                ))
            } else {
                None
            };
            if let Some(reason) = method_reason {
                return block(
                    reason,
                    405,
                    Some(format!("{} not allowed for {}", f.method, f.path)),
                );
            }
            let key = format!("honeypot_get:{}", f.ip);
            if f.method == "GET" {
                if p.honeypot_backend == "memory" {
                    py.allow_threads(|| -> PyResult<()> {
                        self.state
                            .lock()
                            .map_err(err)?
                            .honeypot
                            .insert(f.ip.clone(), (f.now, f.now + 300.0));
                        Ok(())
                    })?;
                } else if ["django_cache", "cache"].contains(&p.honeypot_backend.as_str()) {
                    let result = self.honeypot_cache.as_ref().unwrap().call_method1(
                        py,
                        "set_timestamp",
                        (f.ip.clone(), f.now, f.now + 300.0, 300),
                    );
                    if result.is_err() {
                        return Ok((
                            503,
                            json!({"error":"honeypot_cache_unavailable"}),
                            "Honeypot cache unavailable".into(),
                        ));
                    }
                } else {
                    csv::csv_kv_update(
                        py,
                        self.file("pipeline_state.csv"),
                        key,
                        Some(f.now.to_string()),
                        Some(f.now + 300.0),
                        f.now,
                    )?;
                }
            } else if f.method == "POST" {
                let decision = if p.honeypot_backend == "memory" {
                    py.allow_threads(|| -> PyResult<_> {
                        // Read, evaluate and expire atomically: a concurrent GET
                        // must not be removed by an older POST's expiry decision.
                        let mut state = self.state.lock().map_err(err)?;
                        let time = state
                            .honeypot
                            .get(&f.ip)
                            .filter(|(_, expiry)| f.now <= *expiry)
                            .map(|(time, _)| *time);
                        let decision = policy::policy_form(
                            f.now,
                            time,
                            f.path.clone(),
                            p.min_form_time,
                            p.max_page_time,
                        );
                        if time.is_none() || decision.0 == "page_expired" {
                            state.honeypot.remove(&f.ip);
                        }
                        Ok(decision)
                    })?
                } else if ["django_cache", "cache"].contains(&p.honeypot_backend.as_str()) {
                    let time = match self.cached_honeypot_time(py, &f.ip, f.now) {
                        Ok(time) => time,
                        Err(_) => {
                            return Ok((
                                503,
                                json!({"error":"honeypot_cache_unavailable"}),
                                "Honeypot cache unavailable".into(),
                            ));
                        }
                    };
                    // Do not delete a stale snapshot: a concurrent GET may have
                    // refreshed it. Cache TTL owns physical cleanup.
                    policy::policy_form(
                        f.now,
                        time,
                        f.path.clone(),
                        p.min_form_time,
                        p.max_page_time,
                    )
                } else {
                    let time =
                        csv::csv_kv_get(py, self.file("pipeline_state.csv"), key.clone(), f.now)?
                            .and_then(|v| v.parse::<f64>().ok());
                    policy::policy_form(
                        f.now,
                        time,
                        f.path.clone(),
                        p.min_form_time,
                        p.max_page_time,
                    )
                };
                if decision.0 == "page_expired" {
                    if p.honeypot_backend == "csv" {
                        csv::csv_kv_update(
                            py,
                            self.file("pipeline_state.csv"),
                            key,
                            None,
                            None,
                            f.now,
                        )?;
                    }
                    return Ok((
                        409,
                        json!({"error":"page_expired", "message":decision.3, "reload_required":true}),
                        "Page expired".into(),
                    ));
                }
                if decision.0 == "block" {
                    return block(decision.1.unwrap(), 403, None);
                }
            }
        }
        if f.active.contains("uuid_tamper") {
            if let Some(uuid) = &f.uuid {
                if !uuid.is_empty() && !policy::policy_uuid(uuid.clone()) {
                    let score = self.uuid_score(py, &f.ip, f.now, p.uuid_malformed_weight)?;
                    return block(format!("UUID tampering score={score}"), 403, None);
                }
            }
        }
        if f.active.contains("header_validation") {
            if let Some(reason) = py.allow_threads(|| {
                validate_headers_full_default(
                    &f.headers,
                    &p.required_headers,
                    if p.required_headers.is_empty() {
                        0
                    } else {
                        p.header_min_score
                    },
                )
            }) {
                return block(reason, 403, None);
            }
        }
        Ok((0, Value::Null, String::new()))
    }
}

#[pymethods]
impl NativeCSVPipeline {
    #[new]
    #[pyo3(signature = (profile_json, honeypot_cache=None))]
    fn new(
        py: Python<'_>,
        profile_json: String,
        honeypot_cache: Option<PyObject>,
    ) -> PyResult<Self> {
        let mut p: Profile = serde_json::from_str(&profile_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if p.data_dir.is_empty()
            || !["block", "off"].contains(&p.sql_mode.as_str())
            || !(1..=1048576).contains(&p.max_bytes)
            || !["memory", "csv"].contains(&p.rate_backend.as_str())
            || !["memory", "csv", "django_cache", "cache"].contains(&p.honeypot_backend.as_str())
            || !["ip", "ip_path"].contains(&p.rate_key_mode.as_str())
            || p.enabled.iter().any(|s| !STAGES.contains(&s.as_str()))
            || [p.rate_window, p.anomaly_window, p.uuid_window]
                .iter()
                .any(|v| !v.is_finite() || *v < 1.0)
            || [p.min_form_time, p.max_page_time]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
            || p.rate_max < 0
            || p.rate_flood < p.rate_max
            || p.uuid_threshold < 1
            || [
                p.uuid_malformed_weight,
                p.uuid_not_found_weight,
                p.uuid_success_decay,
            ]
            .iter()
            .any(|v| *v < 0)
        {
            return Err(PyValueError::new_err("Invalid native CSV pipeline profile"));
        }
        if ["django_cache", "cache"].contains(&p.honeypot_backend.as_str()) {
            let cache = honeypot_cache.as_ref().ok_or_else(|| {
                PyValueError::new_err("Cache backend requires a honeypot cache adapter")
            })?;
            for method in ["get_timestamp", "set_timestamp"] {
                if !cache.bind(py).getattr(method)?.is_callable() {
                    return Err(PyValueError::new_err("Invalid honeypot cache adapter"));
                }
            }
        } else if honeypot_cache.is_some() {
            return Err(PyValueError::new_err(
                "Cache adapter requires honeypot_backend=cache or django_cache",
            ));
        }
        if p.exempt_paths
            .iter()
            .any(|s| s.contains('*') || s.contains('?') || s.contains('['))
        {
            return Err(PyValueError::new_err(
                "Pipeline exempt_paths accept literal prefixes, not glob patterns",
            ));
        }
        p.geo_allow = p
            .geo_allow
            .iter()
            .map(|s| s.trim().to_uppercase())
            .filter(|s| !s.is_empty())
            .collect();
        p.geo_block = p
            .geo_block
            .iter()
            .map(|s| s.trim().to_uppercase())
            .filter(|s| !s.is_empty())
            .collect();
        let pipeline = Self {
            profile: p,
            state: Mutex::new(State::default()),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            processed: AtomicU64::new(0),
            honeypot_cache,
        };
        for (name, fields) in [
            ("whitelist.csv", strings(&["ip", "added_date"])),
            ("blacklist.csv", strings(&BLOCKLIST_HEADERS)),
            ("keywords.csv", strings(&["keyword", "added_date"])),
            (
                "geo_blocked_countries.csv",
                strings(&["country", "added_date"]),
            ),
            (
                "path_exemptions.csv",
                strings(&["path", "reason", "added_date"]),
            ),
            (
                "pipeline_state.csv",
                strings(&["key", "value", "expires_at"]),
            ),
            ("rate_buckets.csv", strings(&["key", "value", "expires_at"])),
        ] {
            if name == "pipeline_state.csv" && pipeline.profile.honeypot_backend != "csv" {
                continue;
            }
            let path = pipeline.file(name);
            if !Path::new(&path).exists() {
                csv::csv_write(py, path.clone(), fields.clone(), vec![], true)?;
            }
            if !csv::csv_headers_match(py, path, fields)? {
                return Err(PyValueError::new_err(format!("Invalid CSV schema: {name}")));
            }
        }
        if pipeline.profile.enabled.contains("logging") {
            let path = if pipeline.profile.log_path.is_empty() {
                pipeline.file("requests.csv")
            } else {
                pipeline.profile.log_path.clone()
            };
            if !Path::new(&path).exists() {
                csv::csv_write(py, path.clone(), strings(&LOG_HEADERS), vec![], true)?;
            }
            if !csv::csv_headers_match(py, path, strings(&LOG_HEADERS))? {
                return Err(PyValueError::new_err("Invalid pipeline log CSV schema"));
            }
        }
        Ok(pipeline)
    }
    #[getter]
    fn processed_requests(&self) -> u64 {
        self.processed.load(Ordering::Relaxed)
    }

    #[pyo3(signature = (model=None))]
    fn set_model(&self, model: Option<PyRef<'_, crate::IsolationForest>>) -> PyResult<()> {
        if let Some(m) = &model {
            let state = m.inner.to_state();
            if state.n_features_in_ != 6 || state.trees.is_empty() {
                return Err(PyValueError::new_err(
                    "Pipeline model must be fitted on six request features",
                ));
            }
        }
        self.state.lock().map_err(err)?.model = model.map(|m| Arc::new(m.inner.clone()));
        Ok(())
    }

    /// Explicit warm-start API; eligibility is a framework fact supplied once.
    fn seed_history(
        &self,
        ip: String,
        entries: Vec<(f64, String, i32, f64, bool)>,
    ) -> PyResult<()> {
        if entries.iter().any(|(t, _, s, rt, _)| {
            !t.is_finite() || !rt.is_finite() || *rt < 0.0 || !(100..=599).contains(s)
        }) {
            return Err(PyValueError::new_err("Invalid anomaly history"));
        }
        self.state.lock().map_err(err)?.histories.insert(
            ip,
            entries
                .into_iter()
                .map(|(timestamp, path, status, response_time, kw_check)| {
                    (
                        RecentEntryInput {
                            timestamp,
                            path_lower: path.to_lowercase(),
                            status,
                            kw_check,
                        },
                        response_time,
                    )
                })
                .collect(),
        );
        Ok(())
    }

    fn diagnostics(&self) -> PyResult<(bool, usize, usize, usize)> {
        let state = self.state.lock().map_err(err)?;
        Ok((
            state.model.is_some(),
            state.rates.len(),
            state.uuid.len(),
            state.histories.len(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (ip, path, method, headers, query, body, exists, accepts_get, accepts_post, accepts_method, authenticated, disabled, exempt, now, date, exhausted=false))]
    fn begin(
        &self,
        py: Python<'_>,
        ip: String,
        path: String,
        method: String,
        headers: HashMap<String, String>,
        query: Vec<u8>,
        body: Vec<u8>,
        exists: bool,
        accepts_get: bool,
        accepts_post: bool,
        accepts_method: bool,
        authenticated: bool,
        disabled: HashSet<String>,
        exempt: bool,
        now: f64,
        date: String,
        exhausted: bool,
    ) -> PyResult<(NativePipelineRequest, i32, String)> {
        if !now.is_finite() {
            return Err(PyValueError::new_err("Request time must be finite"));
        }
        let started = std::time::Instant::now();
        let exempt = exempt || self.exempt(py, &ip, &path.to_lowercase())?;
        let active = self
            .profile
            .enabled
            .iter()
            .filter(|s| !disabled.contains(*s) && (!exempt || s.as_str() == "logging"))
            .cloned()
            .collect();
        let uuid = query.split(|b| *b == b'&').find_map(|part| {
            let mut pair = part.splitn(2, |b| *b == b'=');
            (decode(pair.next().unwrap_or_default()) == "uuid")
                .then(|| decode(pair.next().unwrap_or_default()))
        });
        let f = Facts {
            ip,
            path,
            method: method.to_uppercase(),
            headers,
            exists,
            now,
            date,
            active,
            uuid,
            exempt,
            query: String::from_utf8_lossy(&query).to_lowercase(),
        };
        // Bound state maps without evicting active subjects or changing thresholds.
        py.allow_threads(|| -> PyResult<()> {
            let mut state = self.state.lock().map_err(err)?;
            if now >= state.cleanup_at || now < state.cleanup_at - 30.0 {
                state.rates.retain(|_, entries| {
                    entries.iter().any(|t| now - t < self.profile.rate_window)
                });
                state.honeypot.retain(|_, (_, expiry)| now <= *expiry);
                state.uuid.retain(|_, entries| {
                    entries
                        .iter()
                        .any(|(t, _)| now - t <= self.profile.uuid_window)
                });
                state.histories.retain(|_, entries| {
                    entries
                        .iter()
                        .any(|(e, _)| now - e.timestamp < self.profile.anomaly_window)
                });
                state.cleanup_at = now + 30.0;
            }
            Ok(())
        })?;
        let (status, payload, reason) = self.evaluate(
            py,
            &f,
            query,
            body,
            exhausted,
            accepts_get,
            accepts_post,
            accepts_method,
            authenticated,
        )?;
        self.processed.fetch_add(1, Ordering::Relaxed);
        let ctx = NativePipelineRequest {
            owner: self.id,
            facts: f,
            status,
            reason,
            finished: AtomicBool::new(false),
            started,
        };
        Ok((ctx, status, payload.to_string()))
    }

    #[pyo3(signature = (ctx, status, now, content_length, monotonic=false))]
    fn finish(
        &self,
        py: Python<'_>,
        ctx: PyRef<'_, NativePipelineRequest>,
        status: i32,
        now: f64,
        content_length: String,
        monotonic: bool,
    ) -> PyResult<(i32, String)> {
        if ctx.owner != self.id || !now.is_finite() || !(100..=599).contains(&status) {
            return Err(PyValueError::new_err("Invalid pipeline response context"));
        }
        if ctx.finished.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err(
                "Pipeline response already finalized",
            ));
        }
        let elapsed = if monotonic {
            ctx.started.elapsed().as_secs_f64()
        } else {
            (now - ctx.facts.now).max(0.0)
        };
        let mut f = ctx.facts.clone();
        f.now = now;
        let p = &self.profile;
        let mut final_status = status;
        let mut reason = ctx.reason.clone();
        let mut payload = Value::Null;
        if ctx.status == 0
            && f.active.contains("uuid_tamper")
            && f.uuid
                .as_ref()
                .is_some_and(|s| policy::policy_uuid(s.clone()))
        {
            let delta = if status == 404 {
                p.uuid_not_found_weight
            } else if status < 400 {
                -p.uuid_success_decay
            } else {
                0
            };
            let score = self.uuid_score(py, &f.ip, now, delta)?;
            if status == 404 && score >= p.uuid_threshold {
                reason = format!("UUID tampering score={score}");
                self.blocklist(py, &f, &reason)?;
                final_status = 403;
                payload = json!({"error":"blocked"});
            }
        }
        if ctx.status == 0 && f.active.contains("ai_anomaly") {
            let anomaly = py.allow_threads(|| -> PyResult<Option<String>> {
                let mut state = self.state.lock().map_err(err)?;
                let model = state.model.clone();
                let entries = state.histories.entry(f.ip.clone()).or_default();
                entries.retain(|(e, _)| now - e.timestamp < p.anomaly_window);
                let kw_check = !f.exists && !f.exempt;
                let lower = f.path.to_lowercase();
                let hits = if kw_check { p.anomaly_keywords.iter().filter(|s| !s.is_empty() && lower.contains(s.as_str())).count() } else { 0 };
                let statuses = [200,201,204,301,302,400,401,403,404,405,500,502,503];
                let features = vec![f.path.chars().count() as f64, hits as f64, elapsed,
                    statuses.iter().position(|s| *s == final_status).map(|i| i as f64).unwrap_or(-1.0),
                    entries.iter().filter(|(e, _)| now - e.timestamp <= 10.0).count() as f64,
                    entries.iter().filter(|(e, _)| e.status == 404).count() as f64];
                let mut block_reason = None;
                if let Some(model) = model {
                    let prediction = model.predict(&[features])[0];
                    if prediction == -1 {
                        if let Some(stats) = analyze_recent_behavior(entries.iter().map(|(e, _)| e.clone()).collect(), p.anomaly_keywords.clone()) {
                            if stats.should_block { block_reason = Some(format!("AI anomaly + scanning behavior (404s:{}, scanning:{}, kw:{:.1}, burst:{:.1})", stats.max_404s, stats.scanning_404s, stats.avg_kw_hits, stats.avg_burst)); }
                        }
                    } else {
                        let scanning = analyze_recent_behavior(vec![RecentEntryInput { path_lower: lower.clone(), timestamp: now, status: 404, kw_check }], vec![]).is_some_and(|s| s.scanning_404s > 0);
                        let hits = p.anomaly_keywords.iter().filter(|s| !s.is_empty() && lower.contains(s.as_str())).count();
                        if scanning && hits >= 3 { block_reason = Some(format!("AI anomaly + scanning behavior (kw:{hits}, scanning_path:{})", f.path)); }
                    }
                }
                entries.push((RecentEntryInput { path_lower: lower, timestamp: now, status: final_status, kw_check }, elapsed));
                Ok(block_reason)
            })?;
            if let Some(block_reason) = anomaly {
                self.blocklist(py, &f, &block_reason)?;
                if p.ip_blocking {
                    final_status = 403;
                    payload = json!({"error":"blocked"});
                    reason = block_reason;
                }
            }
            if p.learning && status == 404 && !f.exists && !f.exempt {
                let lower = f.path.to_lowercase();
                let suspicious = [
                    "../",
                    "..\\",
                    ".env",
                    "wp-admin",
                    "phpmyadmin",
                    "config",
                    "backup",
                    "database",
                    "mysql",
                    "passwd",
                    "shadow",
                    "admin",
                    "shell",
                    "cmd",
                    "exec",
                    "system",
                    "%2e%2e",
                    "%252e",
                    "%c0%ae",
                    "%2f",
                    "%5c",
                ]
                .iter()
                .any(|s| lower.contains(s))
                    || [
                        "union select",
                        "drop table",
                        "insert into",
                        "delete from",
                        "update set",
                        "<script",
                        "javascript:",
                        "onload=",
                        "onerror=",
                        "document.cookie",
                    ]
                    .iter()
                    .any(|s| f.query.contains(s))
                    || f.query.split('&').any(|part| {
                        ["cmd", "exec", "system", "shell", "eval"].contains(
                            &decode(part.split('=').next().unwrap_or("").as_bytes()).as_str(),
                        )
                    });
                if suspicious {
                    static SEGMENTS: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
                    let learned = SEGMENTS
                        .get_or_init(|| regex::Regex::new(r"\W+").unwrap())
                        .split(&lower)
                        .filter(|s| {
                            s.chars().count() > 3
                                && !p.anomaly_keywords.iter().any(|kw| kw == s)
                                && !p.static_keywords.contains(*s)
                                && !p.legitimate_keywords.contains(*s)
                                && !p.exempt_keywords.contains(*s)
                        })
                        .map(str::to_owned)
                        .collect::<Vec<_>>();
                    if !learned.is_empty() {
                        csv::csv_keywords_update(
                            py,
                            self.file("keywords.csv"),
                            f.date.clone(),
                            learned,
                            HashSet::new(),
                        )?;
                    }
                }
            }
        }
        if f.active.contains("logging") {
            let path = if p.log_path.is_empty() {
                self.file("requests.csv")
            } else {
                p.log_path.clone()
            };
            let length = if payload.is_null() {
                content_length
            } else {
                payload.to_string().len().to_string()
            };
            csv::csv_log_append(
                py,
                path,
                strings(&LOG_HEADERS),
                vec![
                    f.date,
                    f.ip,
                    f.method,
                    f.path,
                    final_status.to_string(),
                    length,
                    (elapsed * 1000.0).trunc().to_string(),
                    f.headers.get("HTTP_REFERER").cloned().unwrap_or_default(),
                    f.headers
                        .get("HTTP_USER_AGENT")
                        .cloned()
                        .unwrap_or_default(),
                    if reason.is_empty() { "False" } else { "True" }.into(),
                    reason,
                ],
                p.log_fsync,
            )?;
        }
        Ok((final_status, payload.to_string()))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeCSVPipeline>()?;
    m.add_class::<NativePipelineRequest>()?;
    Ok(())
}
