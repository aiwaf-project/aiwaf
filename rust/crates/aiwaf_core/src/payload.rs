//! Request payload decoding and the SQL injection check shared by every Rust caller.
//!
//! Python, the CSV pipeline and the request engine all call `inspect_sql`, so there is one
//! implementation of the traversal and its limits.

use serde::Deserialize;
use std::collections::HashMap;

/// Decodes %XX and + as a query string does, keeping invalid escapes as they are.
pub fn decode(bytes: &[u8]) -> String {
    decode_with(bytes, true)
}

/// Decodes %XX only, keeping a literal + as it is.
pub fn decode_percent(bytes: &[u8]) -> String {
    decode_with(bytes, false)
}

fn decode_with(bytes: &[u8], plus_is_space: bool) -> String {
    String::from_utf8_lossy(&decode_bytes(bytes, plus_is_space)).into_owned()
}

/// Decodes %XX (and + when plus_is_space) to bytes, so a binary value isn't lost to UTF-8.
pub fn decode_bytes(bytes: &[u8], plus_is_space: bool) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // IIS-style %uXXXX escapes, as in %u2019 OR 1=1.
        if bytes[i] == b'%' && (bytes.get(i + 1) == Some(&b'u') || bytes.get(i + 1) == Some(&b'U')) && i + 5 < bytes.len() {
            let unit = std::str::from_utf8(&bytes[i + 2..i + 6]).ok().and_then(|s| u32::from_str_radix(s, 16).ok());
            if let Some(c) = unit.and_then(char::from_u32) {
                let mut buffer = [0u8; 4];
                output.extend_from_slice(c.encode_utf8(&mut buffer).as_bytes());
                i += 6;
                continue;
            }
        }
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(a), Some(b)) = (
                (bytes[i + 1] as char).to_digit(16),
                (bytes[i + 2] as char).to_digit(16),
            ) {
                output.push((a * 16 + b) as u8);
                i += 3;
                continue;
            }
        }
        output.push(if plus_is_space && bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    output
}

/// Reads a body as text, detecting UTF-16 and UTF-32 the way Python's json module does.
pub fn body_text(bytes: &[u8]) -> String {
    if bytes.starts_with(&[0xff, 0xfe, 0, 0])
        || bytes.starts_with(&[0, 0, 0xfe, 0xff])
        || (bytes.len() >= 4
            && ((bytes[0] == 0 && bytes[1] == 0)
                || (bytes[1] == 0 && bytes[2] == 0 && bytes[3] == 0)))
    {
        let little = bytes.starts_with(&[0xff, 0xfe]) || bytes[0] != 0;
        let bytes =
            if bytes.starts_with(&[0xff, 0xfe, 0, 0]) || bytes.starts_with(&[0, 0, 0xfe, 0xff]) {
                &bytes[4..]
            } else {
                bytes
            };
        return bytes
            .chunks_exact(4)
            .map(|b| {
                char::from_u32(if little {
                    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
                } else {
                    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
                })
                .unwrap_or('\u{fffd}')
            })
            .collect();
    }
    if bytes.starts_with(&[0xff, 0xfe]) || bytes.starts_with(&[0xfe, 0xff]) {
        let little = bytes[0] == 0xff;
        let words: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|b| {
                if little {
                    u16::from_le_bytes([b[0], b[1]])
                } else {
                    u16::from_be_bytes([b[0], b[1]])
                }
            })
            .collect();
        return String::from_utf16_lossy(&words);
    }
    if bytes.len() >= 2 && (bytes[0] == 0 || bytes[1] == 0) {
        let little = bytes[0] != 0;
        let words: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|b| {
                if little {
                    u16::from_le_bytes([b[0], b[1]])
                } else {
                    u16::from_be_bytes([b[0], b[1]])
                }
            })
            .collect();
        return String::from_utf16_lossy(&words);
    }
    String::from_utf8_lossy(bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes)).into_owned()
}

// Preserve JSON insertion order so reverse traversal has the same finding
// precedence as Python. Map entries are not converted into a sorted Value map.
pub enum JsonTree {
    Scalar,
    Text(String),
    Array(Vec<JsonTree>),
    Object(Vec<(String, JsonTree)>),
}
impl<'de> Deserialize<'de> for JsonTree {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TreeVisitor;
        impl<'de> serde::de::Visitor<'de> for TreeVisitor {
            type Value = JsonTree;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("JSON value")
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<JsonTree, E> {
                Ok(JsonTree::Scalar)
            }
            fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<JsonTree, E> {
                Ok(JsonTree::Scalar)
            }
            fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<JsonTree, E> {
                Ok(JsonTree::Scalar)
            }
            fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<JsonTree, E> {
                Ok(JsonTree::Scalar)
            }
            fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<JsonTree, E> {
                Ok(JsonTree::Scalar)
            }
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<JsonTree, E> {
                Ok(JsonTree::Text(value.into()))
            }
            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<JsonTree, E> {
                Ok(JsonTree::Text(value))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<JsonTree, A::Error> {
                let mut values = vec![];
                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }
                Ok(JsonTree::Array(values))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<JsonTree, A::Error> {
                let mut values: Vec<(String, JsonTree)> = vec![];
                let mut indices: HashMap<String, usize> = HashMap::new();
                while let Some((key, value)) = map.next_entry::<String, JsonTree>()? {
                    // Python retains the original key position but the last value.
                    if let Some(index) = indices.get(&key) {
                        values[*index] = (key, value);
                    } else {
                        indices.insert(key.clone(), values.len());
                        values.push((key, value));
                    }
                }
                Ok(JsonTree::Object(values))
            }
        }
        deserializer.deserialize_any(TreeVisitor)
    }
}

/// Python JSON accepts nonfinite constants and unpaired Unicode surrogates.
/// Normalize only those tokens; never rewrite literal text inside a JSON string.
pub fn parse_json(text: &str) -> Result<Option<JsonTree>, ()> {
    let raw = text.as_bytes();
    let mut normalized = Vec::with_capacity(raw.len());
    let (mut i, mut quoted) = (0, false);
    while i < raw.len() {
        if raw[i] == b'"' {
            quoted = !quoted;
            normalized.push(raw[i]);
            i += 1;
            continue;
        }
        if quoted && raw[i] == b'\\' {
            if raw.get(i + 1) == Some(&b'u') && i + 6 <= raw.len() {
                let word = std::str::from_utf8(&raw[i + 2..i + 6])
                    .ok()
                    .and_then(|s| u16::from_str_radix(s, 16).ok());
                if word.is_some_and(|w| (0xd800..=0xdbff).contains(&w))
                    && raw.get(i + 6..i + 8) == Some(b"\\u")
                    && i + 12 <= raw.len()
                {
                    let low = std::str::from_utf8(&raw[i + 8..i + 12])
                        .ok()
                        .and_then(|s| u16::from_str_radix(s, 16).ok());
                    if low.is_some_and(|w| (0xdc00..=0xdfff).contains(&w)) {
                        normalized.extend_from_slice(&raw[i..i + 12]);
                        i += 12;
                        continue;
                    }
                }
                if word.is_some_and(|w| (0xd800..=0xdfff).contains(&w)) {
                    normalized.extend_from_slice(b"\\ufffd");
                    i += 6;
                    continue;
                }
            }
            // Escaped quotes and escaped backslashes do not change string state.
            normalized.push(raw[i]);
            i += 1;
            if i < raw.len() {
                normalized.push(raw[i]);
                i += 1;
            }
            continue;
        }
        if !quoted {
            let previous_boundary = i == 0 || b" \t\r\n[:,".contains(&raw[i - 1]);
            if previous_boundary {
                // Numeric scalars are not SQL inputs. Python permits large
                // integers and overflowed floats; avoid serde's f64 limit.
                if raw[i] == b'-' || raw[i].is_ascii_digit() {
                    let end = raw[i..]
                        .iter()
                        .position(|b| b" \t\r\n,]}".contains(b))
                        .map(|n| i + n)
                        .unwrap_or(raw.len());
                    static NUMBER: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
                    if NUMBER
                        .get_or_init(|| {
                            regex::Regex::new(
                                r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$",
                            )
                            .unwrap()
                        })
                        .is_match(std::str::from_utf8(&raw[i..end]).unwrap_or(""))
                    {
                        normalized.extend_from_slice(b"null");
                        i = end;
                        continue;
                    }
                }
                if let Some(token) =
                    [b"-Infinity".as_slice(), b"Infinity", b"NaN"]
                        .iter()
                        .find(|token| {
                            raw[i..].starts_with(token)
                                && (i + token.len() == raw.len()
                                    || b" \t\r\n,]}".contains(&raw[i + token.len()]))
                        })
                {
                    normalized.extend_from_slice(b"null");
                    i += token.len();
                    continue;
                }
            }
        }
        normalized.push(raw[i]);
        i += 1;
    }
    match serde_json::from_slice(&normalized) {
        Ok(value) => Ok(Some(value)),
        Err(e) if e.to_string().contains("recursion limit exceeded") => Err(()),
        Err(_) => Ok(None),
    }
}

/// Headers that hold credentials, framing or browser-set values, which aren't checked. Any
/// other header is: apps store and query headers such as X-Forwarded-For, Accept-Language or
/// their own X- headers.
const UNCHECKED_HEADERS: &[&str] = &[
    "accept", "accept-encoding", "authorization", "proxy-authorization", "cache-control", "connection",
    "content-encoding", "content-length", "content-md5", "content-type", "host", "keep-alive", "pragma",
    "te", "trailer", "transfer-encoding", "upgrade", "upgrade-insecure-requests", "dnt", "priority",
    "x-csrf-token", "x-csrftoken", "x-xsrf-token", "x-api-key", "x-amz-security-token", "x-amz-date",
    "x-amz-content-sha256", "x-hub-signature", "x-hub-signature-256", "x-request-id", "x-correlation-id",
    "traceparent", "tracestate", "baggage",
];

fn header_checked(name: &str) -> bool {
    !(UNCHECKED_HEADERS.contains(&name)
        || name.starts_with("sec-")
        || name.starts_with("if-")
        || name.starts_with("content-")
        || name.starts_with("access-control-")
        || name.starts_with(':'))
}

/// Checks a URL path for SQL injection, one segment at a time, as in /post/1 OR 1=1. A query
/// string is ignored: it's checked on its own. The whole path is also checked when a comment
/// spans segments, as in /get/sleep(/*x*/12).
pub fn inspect_sql_path(path: &str, budget: usize) -> Option<(String, i32)> {
    let path = path.split(['?', '#']).next().unwrap_or_default();
    if path.len() > budget {
        return Some(("payload_inspection_limit".into(), 413));
    }
    let check = |segment: &str| inspect_sql_value(segment, budget);
    let found = path.split('/').filter(|segment| !segment.is_empty()).take(256).find_map(|segment| {
        // A matrix parameter, as in /cart;jsessionid=1, is checked by its value.
        check(segment).or_else(|| segment.split_once('=').and_then(|(_, value)| check(value)))
    });
    if found.is_some() {
        return found;
    }
    let decoded = decode(decode(path.as_bytes()).as_bytes());
    if decoded.contains("/*") || decoded.contains("*/") {
        return crate::sql::classify(decoded.trim_start_matches('/')).map(|rule| (rule.to_owned(), 403));
    }
    None
}

/// Checks the headers SQL injection reaches through. Cookie names and values, the Referer's
/// query string and path, and the path and query of X-Original-URL and X-Rewrite-URL are read
/// by their parts. Any other header is checked as one value, except those that hold
/// credentials, framing or browser-set values. Header names are matched without regard to case.
/// One part of a multipart form: its header text, field name, file name and content.
pub struct Part {
    pub headers: String,
    pub name: String,
    pub filename: Option<String>,
    pub content: Vec<u8>,
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > haystack.len() {
        return None;
    }
    memchr::memmem::find(&haystack[from..], needle).map(|at| at + from)
}

fn disposition_param(headers: &str, key: &str) -> Option<String> {
    let lower = headers.to_ascii_lowercase();
    let line = lower.lines().position(|line| line.trim_start().starts_with("content-disposition"))?;
    let original = headers.lines().nth(line)?;
    let lowered = original.to_ascii_lowercase();
    let mut at = 0;
    while let Some(found) = lowered[at..].find(key) {
        let start = at + found;
        at = start + key.len();
        // A whole parameter, so name= isn't read from filename=.
        if start > 0 && !matches!(lowered.as_bytes()[start - 1], b';' | b' ' | b'\t') {
            continue;
        }
        let rest = original[at..].trim_start().strip_prefix('=')?.trim_start();
        return Some(match rest.strip_prefix('"') {
            Some(quoted) => quoted.split('"').next().unwrap_or_default().to_owned(),
            None => rest.split(';').next().unwrap_or_default().trim().to_owned(),
        });
    }
    None
}

/// The parts of a multipart/form-data body, found by its opening boundary line, or None when
/// the body isn't one. A body cut short, as a checked prefix is, gives the parts it has.
pub fn multipart_parts(raw: &[u8]) -> Option<Vec<Part>> {
    let start = raw.iter().position(|b| !matches!(b, b'\r' | b'\n'))?;
    let body = &raw[start..];
    if !body.starts_with(b"--") {
        return None;
    }
    let line_end = find(body, b"\n", 0)?;
    let boundary = body[2..line_end].strip_suffix(b"\r").unwrap_or(&body[2..line_end]);
    if boundary.is_empty() || boundary.len() > 200 || boundary.iter().any(|b| *b < 0x20) {
        return None;
    }
    let mut delimiter = b"\n--".to_vec();
    delimiter.extend_from_slice(boundary);
    let mut parts = Vec::new();
    let mut at = line_end + 1;
    while parts.len() < 1024 && at < body.len() {
        let end = find(body, &delimiter, at).unwrap_or(body.len());
        let chunk = &body[at..end];
        let (head, content) = match find(chunk, b"\r\n\r\n", 0) {
            Some(split) => (&chunk[..split], &chunk[split + 4..]),
            None => match find(chunk, b"\n\n", 0) {
                Some(split) => (&chunk[..split], &chunk[split + 2..]),
                None => (chunk, &b""[..]),
            },
        };
        let headers = String::from_utf8_lossy(head).into_owned();
        if parts.is_empty() && !headers.to_ascii_lowercase().contains("content-disposition") {
            return None;
        }
        let content = content.strip_suffix(b"\r").unwrap_or(content).to_vec();
        parts.push(Part {
            name: disposition_param(&headers, "name").unwrap_or_default(),
            filename: disposition_param(&headers, "filename*").or_else(|| disposition_param(&headers, "filename")),
            headers,
            content,
        });
        // The closing delimiter, --boundary--, ends the form.
        if end >= body.len() || body[end + delimiter.len()..].starts_with(b"--") {
            break;
        }
        at = find(body, b"\n", end + delimiter.len()).map_or(body.len(), |n| n + 1);
    }
    (!parts.is_empty()).then_some(parts)
}

/// The rule for a part's shape: its headers, as in filename="shell.php", its field name, as in
/// __proto__[x], and an uploaded file's content. A text field's value is checked by the caller.
pub fn part_rule(part: &Part) -> Option<&'static str> {
    if crate::rce::polluting_key(&part.name) {
        return Some("rce_prototype_pollution");
    }
    crate::rce::classify(&part.headers)
        .or_else(|| part.filename.as_ref().and_then(|_| crate::rce::classify_upload(&part.content)))
}

pub fn inspect_sql_headers<'a>(headers: impl IntoIterator<Item = (&'a str, &'a str)>, budget: usize) -> Option<(String, i32)> {
    // Every header is checked, so an attack can't be pushed past the ones that are. A request
    // with more headers, or more header bytes, than can be checked is refused, not let through.
    let mut total = 0usize;
    for (index, (name, value)) in headers.into_iter().enumerate() {
        total += name.len() + value.len();
        if index >= 256 || total > budget.max(65536) {
            return Some(("payload_inspection_limit".into(), 413));
        }
        // Names can also be WSGI keys, as in HTTP_USER_AGENT.
        let name = name.trim().to_ascii_lowercase();
        let name = name.strip_prefix("http_").unwrap_or(&name).replace('_', "-");
        if value.trim().is_empty() {
            continue;
        }
        if !header_checked(&name) {
            // Headers skipped for SQL still carry Log4Shell and Shellshock, as in User-Agent
            // or Accept, so they're checked for remote code execution.
            let decoded = decode(value.as_bytes());
            if let Some(rule) = crate::rce::classify(value).or_else(|| crate::rce::classify(&decoded)) {
                return Some((rule.to_owned(), 403));
            }
            continue;
        }
        let found = match name.as_str() {
            "cookie" if value.split(';').nth(256).is_some() => Some(("payload_inspection_limit".into(), 413)),
            "cookie" => value.split(';').find_map(|pair| {
                let (key, item) = pair.trim().split_once('=').unwrap_or(("", pair.trim()));
                crate::sql::classify(key.trim())
                    .map(|rule| (rule.to_owned(), 403))
                    .or_else(|| inspect_sql(b"", item.trim().as_bytes(), budget, false))
            }),
            "referer" | "x-original-url" | "x-rewrite-url" => {
                let (address, query) = value.split_once('?').unwrap_or((value, ""));
                let query = query.split('#').next().unwrap_or_default();
                // The path after the host, as in https://example.com/post/1 OR 1=1.
                let path = match address.split_once("://") {
                    Some((_, rest)) => rest.find('/').map_or("", |at| &rest[at..]),
                    None => address,
                };
                inspect_sql(query.as_bytes(), b"", budget, false).or_else(|| inspect_sql_path(path, budget))
            }
            _ => inspect_sql_value(value, budget),
        };
        if found.is_some() {
            return found;
        }
    }
    None
}

/// Returns the attribute values and text of an XML document, bounded to 512 values.
fn xml_values(text: &str) -> Vec<String> {
    static VALUES: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pattern = VALUES.get_or_init(|| {
        regex::Regex::new(r#"=\s*"([^"]*)"|=\s*'([^']*)'|>([^<]+)<"#).unwrap()
    });
    pattern
        .captures_iter(text)
        .filter_map(|c| c.get(1).or_else(|| c.get(2)).or_else(|| c.get(3)))
        .map(|m| m.as_str().trim().to_owned())
        .filter(|value| !value.is_empty())
        .take(512)
        .collect()
}

/// Checks a request's query and body for SQL injection with the tokenizer.
///
/// Bounded traversal, including JSON keys, escaped strings and repeated URL decoding. Returns
/// the rule and status: 403 for an attack, 413 when the request is too large to inspect.
pub fn inspect_sql(query: &[u8], body: &[u8], budget: usize, exhausted: bool) -> Option<(String, i32)> {
    walk_sql(query, body, budget, exhausted, true)
}

/// Checks one value, such as a header or a path segment, without splitting it into form
/// fields: in 0 <= a and a <= b, the = isn't a field separator.
fn inspect_sql_value(value: &str, budget: usize) -> Option<(String, i32)> {
    walk_sql(b"", value.as_bytes(), budget, false, false)
}

fn walk_sql(query: &[u8], body: &[u8], budget: usize, exhausted: bool, fields: bool) -> Option<(String, i32)> {
    if exhausted {
        return Some(("payload_inspection_limit".into(), 413));
    }
    enum Node {
        Bytes(Vec<u8>),
        Json(JsonTree),
    }
    let mut stack = vec![
        (Node::Bytes(query.to_vec()), 0),
        (Node::Bytes(body.to_vec()), 0),
    ];
    let mut bytes = 0usize;
    let mut nodes = 0usize;
    while let Some((value, depth)) = stack.pop() {
        nodes += 1;
        if nodes > 4096 || depth > 16 {
            return Some(("payload_inspection_limit".into(), 413));
        }
        match value {
            Node::Bytes(raw) => {
                if bytes + raw.len() > budget {
                    return Some(("payload_inspection_limit".into(), 413));
                }
                // A gzip or deflate body, sent with Content-Encoding, is read inflated. Inflating
                // stops past the budget, so a small body that inflates to gigabytes is refused.
                if raw.starts_with(&[0x1f, 0x8b]) || crate::rce::zlib_header(&raw) {
                    let limit = budget - bytes + 1;
                    if let Some(inflated) = crate::rce::inflate(&raw, limit) {
                        if inflated.len() >= limit {
                            return Some(("payload_inspection_limit".into(), 413));
                        }
                        // Counted, so gzip inside gzip can't be inflated again and again.
                        bytes += raw.len();
                        stack.push((Node::Bytes(inflated), depth));
                        continue;
                    }
                }
                // A multipart form is checked part by part: each part's headers and name, a text
                // field's value as any other value, and a file's content for code and objects.
                if let Some(parts) = multipart_parts(&raw) {
                    for part in parts {
                        if let Some(rule) = part_rule(&part) {
                            return Some((rule.to_owned(), 403));
                        }
                        if part.filename.is_none() {
                            // Deeper than a raw body, so the value isn't split into form fields.
                            stack.push((Node::Bytes(part.content), depth + 2));
                        }
                    }
                    continue;
                }
                // A binary serialized object isn't text, so it's checked before it's read as UTF-8.
                if let Some(rule) = crate::rce::classify_binary(&raw) {
                    return Some((rule.to_owned(), 403));
                }
                if raw.contains(&0)
                    || raw.starts_with(&[0xef, 0xbb, 0xbf])
                    || raw.starts_with(&[0xff, 0xfe])
                    || raw.starts_with(&[0xfe, 0xff])
                {
                    match parse_json(&body_text(&raw)) {
                        Ok(Some(value)) => {
                            bytes += raw.len();
                            stack.push((Node::Json(value), depth + 1));
                            continue;
                        }
                        Err(()) => return Some(("payload_inspection_limit".into(), 413)),
                        _ => (),
                    }
                }
                stack.push((
                    Node::Json(JsonTree::Text(String::from_utf8_lossy(&raw).into_owned())),
                    depth + 1,
                ));
            }
            Node::Json(JsonTree::Text(text)) => {
                bytes += text.len();
                if bytes > budget {
                    return Some(("payload_inspection_limit".into(), 413));
                }
                let decoded = decode(decode(text.as_bytes()).as_bytes());
                // Remote code execution is checked in the whole document first: a gadget such as
                // {"@type": "com.sun.rowset.JdbcRowSetImpl"} or <dynamic-proxy> is in its shape,
                // which checking its values one by one wouldn't see.
                let document = |value: &str| value.trim_start().starts_with(['{', '[', '<']);
                if document(&text) || document(&decoded) {
                    let found = crate::rce::classify_document(&text)
                        .or_else(|| if decoded != text { crate::rce::classify_document(&decoded) } else { None });
                    if let Some(rule) = found {
                        return Some((rule.to_owned(), 403));
                    }
                }
                // A JSON document is checked by its values, so its own quotes aren't read as a
                // breakout. Parse it before decoding when that works, so a literal + survives.
                let mut parsed = false;
                for candidate in [&text, &decoded] {
                    if parsed || !candidate.trim_start().starts_with(['{', '[']) {
                        continue;
                    }
                    match parse_json(candidate) {
                        Ok(Some(value)) => {
                            stack.push((Node::Json(value), depth + 1));
                            parsed = true;
                        }
                        Err(()) => return Some(("payload_inspection_limit".into(), 413)),
                        Ok(None) => (),
                    }
                }
                if !parsed && decoded.trim_start().starts_with('<') {
                    // An XML body is checked by its attribute values and text, as in
                    // <a probe="1234 OR 1=1">, so its own quotes aren't read as a breakout.
                    let values = xml_values(&decoded);
                    if !values.is_empty() {
                        for value in values {
                            stack.push((Node::Json(JsonTree::Text(value)), depth + 1));
                        }
                        parsed = true;
                    }
                }
                if !parsed {
                    // Form decoding turns a literal + into a space, as in '+(SELECT 1)+', so also
                    // check the value with only its %XX escapes decoded.
                    let kept = decode_percent(decode_percent(text.as_bytes()).as_bytes());
                    // Only a raw query string or body is split into form fields. A JSON value,
                    // as in "--warp=e=mc^2", is one value.
                    let check = if fields && depth <= 1 { crate::sql::classify_fields } else { crate::sql::classify };
                    // Remote code execution first: those rules match only attack text.
                    // A form's fields are also checked one by one, for a value that starts with an
                    // attack, as in data=rO0AB..., and for a prototype key, as in __proto__[x]=1.
                    let rule = (if fields && depth <= 1 { crate::rce::classify_fields(&text) } else { None })
                        .or_else(|| crate::rce::classify(&decoded))
                        .or_else(|| if kept != decoded { crate::rce::classify(&kept) } else { None })
                        .or_else(|| check(&decoded))
                        .or_else(|| if kept != decoded { check(&kept) } else { None });
                    if let Some(rule) = rule {
                        return Some((rule.to_owned(), 403));
                    }
                }
            }
            Node::Json(JsonTree::Array(values)) => {
                for value in values {
                    stack.push((Node::Json(value), depth + 1));
                }
            }
            Node::Json(JsonTree::Object(values)) => {
                // {"__proto__": {...}} or {"constructor": {"prototype": {...}}} reaches the prototype
                // of every object when merged into one.
                let polluting = values.iter().any(|(key, value)| {
                    key == "__proto__"
                        || (key == "constructor"
                            && matches!(value, JsonTree::Object(inner) if inner.iter().any(|(k, _)| k == "prototype")))
                });
                if polluting {
                    return Some(("rce_prototype_pollution".into(), 403));
                }
                for (key, value) in values {
                    stack.push((Node::Json(JsonTree::Text(key)), depth + 1));
                    stack.push((Node::Json(value), depth + 1));
                }
            }
            _ => (),
        }
        if stack.len() > 4096 {
            return Some(("payload_inspection_limit".into(), 413));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::inspect_sql;

    fn rule(query: &str, body: &str) -> Option<String> {
        inspect_sql(query.as_bytes(), body.as_bytes(), 65536, false).map(|(rule, _)| rule)
    }

    #[test]
    fn json_bodies_are_checked_by_their_values() {
        assert_eq!(rule("", r#"{"field": "const fn = match(\"/*splat\");"}"#), None);
        assert_eq!(rule("", r#"{"a": "x\" OR 1=1--"}"#).as_deref(), Some("sql_breakout_double_quote"));
        assert_eq!(rule("", r#"["ok", {"b": "1 OR 1=1"}]"#).as_deref(), Some("sql_breakout_numeric"));
    }

    #[test]
    fn a_literal_plus_is_still_seen() {
        assert_eq!(rule("", r#"{"q": "'+(SELECT 1)+'"}"#).as_deref(), Some("sql_breakout_single_quote"));
        assert_eq!(rule("", "'+(SELECT 1)+'").as_deref(), Some("sql_breakout_single_quote"));
    }

    #[test]
    fn query_strings_limits_and_encodings() {
        assert_eq!(rule("id=1%20OR%201%3D1&p=2", "").as_deref(), Some("sql_breakout_numeric"));
        assert_eq!(rule("%7B%22a%22%3A%221%20OR%201%3D1%22%7D", "").as_deref(), Some("sql_breakout_numeric"));
        assert_eq!(inspect_sql(b"", b"aaaaaaaaaaaa", 10, false), Some(("payload_inspection_limit".into(), 413)));
        assert_eq!(inspect_sql(b"", b"", 10, true), Some(("payload_inspection_limit".into(), 413)));
        let utf16: Vec<u8> = "\u{feff}{\"a\":\"x' OR 1=1--\"}".encode_utf16().flat_map(u16::to_le_bytes).collect();
        assert_eq!(inspect_sql(b"", &utf16, 65536, false).map(|(r, _)| r).as_deref(), Some("sql_breakout_single_quote"));
        assert_eq!(rule("q=hello", r#"{"name": "O'Reilly", "n": NaN}"#), None);
    }
}

#[cfg(test)]
mod field_tests {
    use super::inspect_sql;

    #[test]
    fn json_values_are_not_split_into_form_fields() {
        let body = br#"{"cmd": "getopts([\"--warp=e=mc^2\"])"}"#;
        assert_eq!(inspect_sql(b"", body, 65536, false), None);
        assert_eq!(inspect_sql(b"q=Use+%27in%27+or+%27between%27+here", b"", 65536, false), None);
        assert!(inspect_sql(b"", b"a=1&b=1 OR 1=1", 65536, false).is_some());
    }
}

#[cfg(test)]
mod header_tests {
    use super::inspect_sql_headers;

    fn rule(name: &str, value: &str) -> Option<String> {
        inspect_sql_headers([(name, value)], 65536).map(|(rule, _)| rule)
    }

    #[test]
    fn cookies_referers_and_user_agents_are_checked() {
        assert_eq!(rule("Cookie", "theme=dark; session=' OR '1'='x'--").as_deref(), Some("sql_breakout_single_quote"));
        assert_eq!(rule("cookie", "id=1%20OR%201%3D1").as_deref(), Some("sql_breakout_numeric"));
        assert_eq!(rule("Referer", "https://example.com/search?q=1 waitfor delay '0:0:15' --").as_deref(), Some("sql_function"));
        assert_eq!(rule("User-Agent", "1' and lo_get(16400)::text::bool and 'true").as_deref(), Some("sql_breakout_single_quote"));
        assert_eq!(rule("User-Agent", "1%27%20OR%201%3D1--").as_deref(), Some("sql_breakout_single_quote"));
        assert_eq!(rule("HTTP_COOKIE", "a=' OR 1=1--").as_deref(), Some("sql_breakout_single_quote"));
        assert_eq!(rule("HTTP_USER_AGENT", "Mozilla/5.0 (X11; Linux x86_64)"), None);
    }

    #[test]
    fn shared_header_cases() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../../../../tests/fixtures/sql_header_cases.json")).unwrap();
        let pair = |item: &serde_json::Value| (item[0].as_str().unwrap().to_owned(), item[1].as_str().unwrap().to_owned());
        for item in cases["attacks"].as_array().unwrap() {
            let (name, value) = pair(item);
            assert!(rule(&name, &value).is_some(), "missed {name}: {value}");
        }
        for item in cases["ordinary"].as_array().unwrap() {
            let (name, value) = pair(item);
            assert_eq!(rule(&name, &value), None, "flagged {name}: {value}");
        }
    }

    #[test]
    fn ordinary_headers_pass_and_other_headers_are_ignored() {
        for (name, value) in [
            ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"),
            ("User-Agent", "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"),
            ("Cookie", "_ga=GA1.2.1234567890.1700000000; csrftoken=abc123; prefs=%7B%22theme%22%3A%22dark%22%7D"),
            ("Referer", "https://www.example.fr/327-peintures?categories=a,b&page=2"),
            ("Authorization", "Bearer ' OR 1=1--"),
            ("Content-Type", "text/plain; charset=' OR 1=1--"),
            ("Sec-Ch-Ua", "\"Chromium\";v=\"124\", \"Google Chrome\";v=\"124\", \"Not-A.Brand\";v=\"99\""),
            ("If-None-Match", "W/\"' OR 1=1--\""),
            ("X-Forwarded-For", "203.0.113.7, 198.51.100.2"),
            ("Forwarded", "for=192.0.2.60;proto=http;by=203.0.113.43"),
            ("Accept-Language", "en-US,en;q=0.9,fr;q=0.8"),
            ("X-Requested-With", "XMLHttpRequest"),
        ] {
            assert_eq!(rule(name, value), None, "{name}: {value}");
        }
    }

    #[test]
    fn other_headers_are_checked() {
        assert_eq!(rule("X-Anything", "' OR 1=1--").as_deref(), Some("sql_breakout_single_quote"));
        assert_eq!(rule("SomeHeader", "DELETE FROM users;--").as_deref(), Some("sql_breakout_numeric"));
        assert_eq!(rule("X-Forwarded-For", "127.0.0.1' AND SLEEP(5)--").as_deref(), Some("sql_breakout_single_quote"));
        assert_eq!(rule("HTTP_X_REAL_IP", "1' or '1'='1").as_deref(), Some("sql_breakout_single_quote"));
        assert_eq!(rule("X-Original-URL", "/post/1%20OR%201=1").as_deref(), Some("sql_breakout_numeric"));
        assert_eq!(rule("Referer", "https://example.com/post/1%20UNION%20SELECT%20password%20FROM%20users").as_deref(), Some("sql_breakout_numeric"));
    }
}

#[cfg(test)]
mod path_tests {
    use super::inspect_sql_path;

    fn rule(path: &str) -> Option<String> {
        inspect_sql_path(path, 65536).map(|(rule, _)| rule)
    }

    #[test]
    fn payloads_in_the_path_are_found() {
        for path in [
            "/post/1234%20OR%201=1",
            "/post/EmptyValue'%20and%20526=527",
            "/post/foo')waitfor%20delay'5%3a0%3a20'--",
            "/post/sleep(20)",
            "/post/foo/24'union+all+select+1,2,3+from+aa/bar",
            "/get/if(now()=sysdate(),sleep(/*comment*/12),0)/",
            "/items;id=1%20OR%201=1",
        ] {
            assert!(rule(path).is_some(), "missed {path}");
        }
    }

    #[test]
    fn shared_path_cases() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../../../../tests/fixtures/sql_path_cases.json")).unwrap();
        for path in cases["attacks"].as_array().unwrap() {
            assert!(rule(path.as_str().unwrap()).is_some(), "missed {path}");
        }
        for path in cases["ordinary"].as_array().unwrap() {
            assert_eq!(rule(path.as_str().unwrap()), None, "flagged {path}");
        }
    }

    #[test]
    fn ordinary_paths_pass() {
        for path in [
            "/", "/api/v1/orders/123/items", "/users/o'reilly", "/wiki/Don't_Stop_Me_Now", "/blog/2024/01/union-select-committee",
            "/search/rock%20'n'%20roll", "/files/report%20(1).pdf", "/product/12\"-ruler", "/drop/table-of-contents",
            "/cart;jsessionid=8F1A2B3C", "/static/js/app.min.js", "/@user/posts", "/~home/index.html", "/c++/tutorial",
            "/select-a-colour", "/orders?id=1 OR 1=1",
        ] {
            assert_eq!(rule(path), None, "flagged {path}");
        }
    }
}
