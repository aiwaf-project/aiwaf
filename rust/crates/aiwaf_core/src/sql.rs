//! Finds values that break out of a SQL value and continue as SQL.
//!
//! Each value is tried in the three places a request value usually lands in a query: as a bare
//! number, inside single quotes, and inside double quotes. It's flagged only when, in one of
//! them, it closes the value and what follows reads as a SQL condition or statement.
//!
//! This is the main implementation. Python's `aiwaf.core.sql_tokenizer` is the fallback when
//! Rust is turned off, and Node.js and Java have their own copies. The shared cases in
//! `tests/fixtures/sql_injection_cases.json` run against all of them.

use regex::Regex;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Number,
    Str,
    Word,
    Keyword,
    Function,
    Operator,
    Open,
    Close,
    Comma,
    Semicolon,
    Comment,
    Unknown,
    Unterminated,
}

#[derive(Clone, Debug)]
struct Token {
    kind: Kind,
    text: String,
}

const KEYWORDS: &[&str] = &[
    "all", "alter", "and", "as", "asc", "between", "by", "case", "collate", "create", "declare", "delay",
    "delete", "desc", "distinct", "div", "drop", "else", "end", "exec", "execute", "exists", "false",
    "from", "group", "having", "in", "insert", "into", "is", "join", "like", "limit", "not", "null",
    "offset", "on", "or", "order", "outfile", "procedure", "regexp", "rlike", "select", "set", "shutdown",
    "sounds", "table", "then", "true", "truncate", "union", "update", "values", "waitfor", "when",
    "where", "xor",
];
const CONNECTORS: &[&str] = &[
    "and", "between", "collate", "div", "group", "having", "in", "into", "is", "like", "limit", "or",
    "order", "procedure", "regexp", "rlike", "sounds", "union", "xor",
];
// < and > are left out: quoted HTML such as ('<script>') would read as a comparison.
const OPERATOR_CONNECTORS: &[&str] = &["=", "<>", "!=", "<=>", "||", "&&", "+"];
const COMPARISONS: &[&str] = &["=", "<>", "!=", "<", ">", "<=", ">=", "<=>"];
const STATEMENTS: &[&str] = &[
    "alter", "create", "declare", "delete", "drop", "exec", "execute", "insert", "select", "set",
    "shutdown", "truncate", "update", "waitfor",
];
const STRONG_KEYWORDS: &[&str] = &[
    "between", "exists", "group", "having", "in", "into", "is", "like", "limit", "order", "procedure",
    "regexp", "rlike", "select", "union",
];
const VALUE_KEYWORDS: &[&str] = &["false", "null", "true"];
const END_KEYWORDS: &[&str] = &["asc", "desc", "end", "false", "null", "true"];
const MULTI_OPERATORS: &[&str] = &["<=>", "<>", "!=", "<=", ">=", "||", "&&", "::", ":="];
const SINGLE_OPERATORS: &str = "=<>+-*/%&|^~!";
const WHITESPACE: &str = " \t\n\r\u{0b}\u{0c}\u{a0}\u{1680}\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\u{2006}\u{2007}\u{2008}\u{2009}\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}\u{feff}";

/// Keeps MySQL's executed `/*! ... */` comments, drops other block comments, and unifies whitespace.
pub fn clean(value: &str) -> String {
    static VERSIONED: OnceLock<Regex> = OnceLock::new();
    static COMMENT: OnceLock<Regex> = OnceLock::new();
    let text = VERSIONED
        .get_or_init(|| Regex::new(r"(?s)/\*!\d{0,6}(.*?)\*/").unwrap())
        .replace_all(value, " $1 ");
    let text = COMMENT
        .get_or_init(|| Regex::new(r"(?s)/\*.*?\*/").unwrap())
        .replace_all(&text, " ");
    text.chars().map(|c| if WHITESPACE.contains(c) { ' ' } else { c }).collect()
}

fn digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn word_char(c: char) -> bool {
    c.is_alphabetic() || digit(c) || "_$@.".contains(c)
}

fn token(kind: Kind, text: impl Into<String>) -> Token {
    Token { kind, text: text.into() }
}

fn starts_with(chars: &[char], index: usize, text: &str) -> bool {
    let mut i = index;
    for c in text.chars() {
        if chars.get(i) != Some(&c) {
            return false;
        }
        i += 1;
    }
    true
}

fn find(chars: &[char], from: usize, text: &str) -> Option<usize> {
    (from..chars.len()).find(|&i| starts_with(chars, i, text))
}

fn lex(chars: &[char]) -> Vec<Token> {
    let mut tokens = Vec::new();
    let length = chars.len();
    let mut index = 0;
    while index < length {
        let c = chars[index];
        if c == ' ' {
            index += 1;
        } else if c == '\'' || c == '"' || c == '`' {
            let mut end = index + 1;
            while end < length {
                if chars[end] == '\\' && c != '`' {
                    end += 2;
                } else if chars[end] == c && end + 1 < length && chars[end + 1] == c {
                    end += 2;
                } else if chars[end] == c {
                    break;
                } else {
                    end += 1;
                }
            }
            if end >= length {
                tokens.push(token(Kind::Unterminated, ""));
                return tokens;
            }
            let text: String = chars[index + 1..end].iter().collect();
            tokens.push(token(if c == '`' { Kind::Word } else { Kind::Str }, text));
            index = end + 1;
        } else if c == '#' || starts_with(chars, index, "--") || starts_with(chars, index, "/*") {
            let block = c == '/';
            let end = if block { find(chars, index + 2, "*/") } else { find(chars, index, "\n") };
            tokens.push(token(Kind::Comment, ""));
            index = match end {
                None => length,
                Some(end) => end + if block { 2 } else { 1 },
            };
        } else if digit(c) || (c == '.' && index + 1 < length && digit(chars[index + 1])) {
            let mut end = index + 1;
            if c == '0' && end < length && (chars[end] == 'x' || chars[end] == 'X') {
                end += 1;
                while end < length && chars[end].is_ascii_hexdigit() {
                    end += 1;
                }
            } else {
                while end < length && (digit(chars[end]) || chars[end] == '.') {
                    end += 1;
                }
                if end + 1 < length
                    && (chars[end] == 'e' || chars[end] == 'E')
                    && (digit(chars[end + 1]) || chars[end + 1] == '+' || chars[end + 1] == '-')
                {
                    end += 2;
                    while end < length && digit(chars[end]) {
                        end += 1;
                    }
                }
            }
            if end < length && (chars[end].is_alphabetic() || chars[end] == '_') {
                while end < length && word_char(chars[end]) {
                    end += 1;
                }
                tokens.push(token(Kind::Word, chars[index..end].iter().collect::<String>().to_lowercase()));
            } else {
                tokens.push(token(Kind::Number, chars[index..end].iter().collect::<String>()));
            }
            index = end;
        } else if c.is_alphabetic() || "_$@".contains(c) {
            let mut end = index + 1;
            while end < length && word_char(chars[end]) {
                end += 1;
            }
            let word = chars[index..end].iter().collect::<String>().to_lowercase();
            let mut after = end;
            while after < length && chars[after] == ' ' {
                after += 1;
            }
            let kind = if KEYWORDS.contains(&word.as_str()) {
                Kind::Keyword
            } else if after < length && chars[after] == '(' {
                Kind::Function
            } else {
                Kind::Word
            };
            tokens.push(token(kind, word));
            index = end;
        } else if c == '(' {
            tokens.push(token(Kind::Open, "("));
            index += 1;
        } else if c == ')' {
            tokens.push(token(Kind::Close, ")"));
            index += 1;
        } else if c == ',' {
            tokens.push(token(Kind::Comma, ","));
            index += 1;
        } else if c == ';' {
            tokens.push(token(Kind::Semicolon, ";"));
            index += 1;
        } else if let Some(op) = MULTI_OPERATORS.iter().find(|op| starts_with(chars, index, op)) {
            tokens.push(token(Kind::Operator, *op));
            index += op.chars().count();
        } else if SINGLE_OPERATORS.contains(c) {
            tokens.push(token(Kind::Operator, c.to_string()));
            index += 1;
        } else {
            tokens.push(token(Kind::Unknown, c.to_string()));
            index += 1;
        }
    }
    tokens
}

fn is_keyword(token: &Token, words: &[&str]) -> bool {
    token.kind == Kind::Keyword && words.contains(&token.text.as_str())
}

fn is_atom(token: &Token) -> bool {
    matches!(token.kind, Kind::Number | Kind::Str | Kind::Word) || is_keyword(token, VALUE_KEYWORDS)
}

/// Rejects sequences no SQL dialect accepts: adjacent values, open strings, dangling ends.
fn valid(tokens: &[Token]) -> bool {
    let mut last: Option<&Token> = None;
    let mut previous_atom = false;
    for token in tokens {
        if token.kind == Kind::Comment {
            break;
        }
        if matches!(token.kind, Kind::Unknown | Kind::Unterminated) {
            return false;
        }
        if let Some(previous) = last {
            if previous.kind == Kind::Function && token.kind != Kind::Open {
                return false;
            }
        }
        let atom = is_atom(token);
        if atom && previous_atom {
            return false;
        }
        previous_atom = atom || token.kind == Kind::Close;
        last = Some(token);
    }
    match last {
        None => false,
        Some(last) => {
            !matches!(last.kind, Kind::Operator | Kind::Open | Kind::Comma | Kind::Function | Kind::Semicolon)
                && !(last.kind == Kind::Keyword && !END_KEYWORDS.contains(&last.text.as_str()))
        }
    }
}

fn strong(tokens: &[Token]) -> bool {
    tokens.iter().any(|t| {
        matches!(t.kind, Kind::Function | Kind::Comment | Kind::Semicolon)
            || (t.kind == Kind::Operator && COMPARISONS.contains(&t.text.as_str()))
            || is_keyword(t, STRONG_KEYWORDS)
    })
}

/// Checks what follows the end of the value: it must continue the query as SQL.
fn breaks_out(tokens: &[Token], numeric: bool, comment_ok: bool) -> bool {
    let mut index = 0;
    while index < tokens.len() && tokens[index].kind == Kind::Close {
        index += 1;
    }
    let Some(first) = tokens.get(index) else {
        return false;
    };
    let rest = &tokens[index + 1..];
    match first.kind {
        // As in admin'-- : the rest of the query is commented out. With an even number of
        // quotes, as in .hex('#C0FFEE'), the value closes its own string, so it's text.
        Kind::Comment => return !numeric && comment_ok,
        Kind::Semicolon => return rest.first().is_some_and(|t| is_keyword(t, STATEMENTS)),
        Kind::Keyword if CONNECTORS.contains(&first.text.as_str()) => {}
        Kind::Operator if !numeric && OPERATOR_CONNECTORS.contains(&first.text.as_str()) => {}
        _ => return false,
    }
    if first.text == "union" {
        let next = rest
            .iter()
            .find(|t| t.kind != Kind::Open && t.text != "all" && t.text != "distinct");
        if !next.is_some_and(|t| is_keyword(t, &["select"])) {
            return false;
        }
    }
    if (first.text == "order" || first.text == "group") && !rest.first().is_some_and(|t| is_keyword(t, &["by"])) {
        return false;
    }
    // After a bare number, the connector alone isn't enough: "1 like this" is valid SQL too.
    valid(rest) && strong(if numeric { rest } else { &tokens[index..] })
}

fn closing_quote(chars: &[char], quote: char, backslash: bool) -> Option<usize> {
    let mut index = 0;
    while index < chars.len() {
        let c = chars[index];
        if backslash && c == '\\' {
            index += 2;
        } else if c == quote && index + 1 < chars.len() && chars[index + 1] == quote {
            index += 2;
        } else if c == quote {
            return Some(index);
        } else {
            index += 1;
        }
    }
    None
}

/// True for a quoted SQL word in prose or code, as in `.method('is')` or `with "in" or "between"`.
/// The quote opens after a space or bracket and closes straight after one keyword.
fn quoted_word(chars: &[char], end: usize, quote: char) -> bool {
    // + is a space in form-encoded text, as in Use+'in'+or+'between'.
    if end == 0 || !" +([{,:=".contains(chars[end - 1]) {
        return false;
    }
    let mut stop = end + 1;
    while stop < chars.len() && chars[stop].is_alphabetic() {
        stop += 1;
    }
    stop < chars.len()
        && chars[stop] == quote
        && KEYWORDS.contains(&chars[end + 1..stop].iter().collect::<String>().to_lowercase().as_str())
}

/// Returns a rule name when the value breaks out of a SQL value, otherwise `None`.
pub fn classify(value: &str) -> Option<&'static str> {
    let chars: Vec<char> = clean(value).chars().collect();
    let tokens = lex(&chars);
    let start = usize::from(
        tokens.len() > 1 && tokens[0].kind == Kind::Operator && (tokens[0].text == "-" || tokens[0].text == "+"),
    );
    if tokens.get(start).is_some_and(|t| t.kind == Kind::Number) && breaks_out(&tokens[start + 1..], true, true) {
        return Some("sql_breakout_numeric");
    }
    for (quote, rule) in [('\'', "sql_breakout_single_quote"), ('"', "sql_breakout_double_quote")] {
        if !chars.contains(&quote) {
            continue;
        }
        // MySQL treats a backslash as an escape and most other databases don't, so try both.
        let modes: &[bool] = if chars.contains(&'\\') { &[true, false] } else { &[false] };
        for &backslash in modes {
            if let Some(end) = closing_quote(&chars, quote, backslash).filter(|&end| !quoted_word(&chars, end, quote)) {
                // The query's own closing quote follows the value.
                let mut tail: Vec<char> = chars[end + 1..].to_vec();
                tail.push(quote);
                let odd = chars.iter().filter(|&&c| c == quote).count() % 2 == 1;
                if breaks_out(&lex(&tail), false, odd) {
                    return Some(rule);
                }
            }
        }
    }
    None
}

/// Checks a value, and each name and value in it when it's a raw query string or form body.
pub fn classify_fields(value: &str) -> Option<&'static str> {
    if let Some(rule) = classify(value) {
        return Some(rule);
    }
    if !value.contains('=') {
        return None;
    }
    value.split('&').take(512).find_map(|part| match part.split_once('=') {
        Some((name, field)) => classify(name).or_else(|| classify(field)),
        None => classify(part),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cases() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../../tests/fixtures/sql_injection_cases.json")).unwrap()
    }

    #[test]
    fn every_shared_attack_is_caught() {
        for value in cases()["attacks"].as_array().unwrap() {
            let value = value.as_str().unwrap();
            assert!(classify(value).is_some(), "missed {value:?}");
        }
    }

    #[test]
    fn no_shared_ordinary_value_is_flagged() {
        for value in cases()["ordinary"].as_array().unwrap() {
            let value = value.as_str().unwrap();
            assert_eq!(classify(value), None, "flagged {value:?}");
        }
    }

    #[test]
    fn ordinary_text_is_never_flagged() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../../../../tests/fixtures/ordinary_text.json")).unwrap();
        let flagged: Vec<&str> = cases["values"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|value| value.as_str())
            .filter(|value| classify_fields(value).is_some())
            .collect();
        assert!(flagged.is_empty(), "flagged {flagged:?}");
    }

    #[test]
    fn each_position_has_its_own_rule() {
        assert_eq!(classify("1 OR 1=1"), Some("sql_breakout_numeric"));
        assert_eq!(classify("x' OR 1=1--"), Some("sql_breakout_single_quote"));
        assert_eq!(classify("x\" OR 1=1--"), Some("sql_breakout_double_quote"));
        assert_eq!(classify(""), None);
    }

    #[test]
    fn cleanup_keeps_executed_comments_and_unifies_whitespace() {
        assert_eq!(clean("1 /*!50000UNION*/ SELECT"), "1  UNION  SELECT");
        assert_eq!(clean("a/**/b"), "a b");
        assert_eq!(clean("a\u{0b}b\u{a0}c"), "a b c");
    }

    #[test]
    fn raw_query_strings_are_checked_field_by_field() {
        assert_eq!(classify_fields("id=1 OR 1=1&page=2"), Some("sql_breakout_numeric"));
        assert_eq!(classify_fields("name=O'Neil&city=x"), None);
        assert_eq!(classify_fields("plain"), None);
    }

    #[test]
    fn long_values_finish() {
        let value = format!("{}' OR 1=1--", "a".repeat(100_000));
        assert_eq!(classify(&value), Some("sql_breakout_single_quote"));
        assert_eq!(classify(&"'".repeat(10_001)), None);
    }
}
