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
    let mut output = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
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
    String::from_utf8_lossy(&output).into_owned()
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

/// Checks a request's query and body for SQL injection with the tokenizer.
///
/// Bounded traversal, including JSON keys, escaped strings and repeated URL decoding. Returns
/// the rule and status: 403 for an attack, 413 when the request is too large to inspect.
pub fn inspect_sql(query: &[u8], body: &[u8], budget: usize, exhausted: bool) -> Option<(String, i32)> {
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
                if !parsed {
                    // Form decoding turns a literal + into a space, as in '+(SELECT 1)+', so also
                    // check the value with only its %XX escapes decoded.
                    let kept = decode_percent(decode_percent(text.as_bytes()).as_bytes());
                    // Only a raw query string or body is split into form fields. A JSON value,
                    // as in "--warp=e=mc^2", is one value.
                    let check = if depth <= 1 { crate::sql::classify_fields } else { crate::sql::classify };
                    let rule = check(&decoded).or_else(|| if kept != decoded { check(&kept) } else { None });
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
