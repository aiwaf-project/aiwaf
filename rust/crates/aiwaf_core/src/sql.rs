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
    "order", "procedure", "regexp", "rlike", "sounds", "union", "where", "xor",
];
// MSSQL runs a second statement without a semicolon, as in ' WAITFOR DELAY '0:0:5'.
const BARE_STATEMENTS: &[&str] = &["declare", "exec", "execute", "waitfor"];
// < and > are left out: quoted HTML such as ('<script>') would read as a comparison.
const OPERATOR_CONNECTORS: &[&str] = &["=", "<>", "!=", "<=>", "||", "&&", "+"];
const COMPARISONS: &[&str] = &["=", "<>", "!=", "<", ">", "<=", ">=", "<=>"];
const STATEMENTS: &[&str] = &[
    "alter", "create", "declare", "delete", "drop", "exec", "execute", "insert", "select", "set",
    "shutdown", "truncate", "update", "waitfor",
];
const STRONG_KEYWORDS: &[&str] = &[
    "between", "exists", "group", "having", "in", "into", "is", "like", "limit", "order", "procedure",
    "regexp", "rlike", "select",
];
const VALUE_KEYWORDS: &[&str] = &["false", "null", "true"];
const END_KEYWORDS: &[&str] = &["asc", "desc", "end", "false", "null", "true"];
const MULTI_OPERATORS: &[&str] = &["<=>", "<>", "!=", "<=", ">=", "||", "&&", "::", ":="];
const SINGLE_OPERATORS: &str = "=<>+-*/%&|^~!";
// A null byte counts as a space: sqlmap ends payloads with one, as in -8970) ORDER BY 3631%00.
const WHITESPACE: &str = "\0 \t\n\r\u{0b}\u{0c}\u{a0}\u{1680}\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\u{2006}\u{2007}\u{2008}\u{2009}\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}\u{feff}";

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
    text.chars()
        .map(|c| match c {
            c if WHITESPACE.contains(c) => ' ',
            // Apps often normalise typographic quotes to ', as in %u2019 OR 1=1-- -.
            '\u{2018}' | '\u{2019}' | '\u{02bc}' | '\u{ff07}' => '\'',
            c => c,
        })
        .collect()
}

/// Names that only appear in SQL, as in pg_sleep( or information_schema. They're flagged
/// wherever they appear, because ordinary text doesn't contain them.
fn sql_only_name(text: &str) -> bool {
    static NAMES: OnceLock<Regex> = OnceLock::new();
    NAMES
        .get_or_init(|| {
            Regex::new(concat!(
                r"(?i)(?-u:\b)(?:pg_sleep|load_file|extractvalue|updatexml|dbms_pipe\.\w+|utl_inaddr\.\w+|utl_http\.\w+|sys_eval|sys_exec)\s*\(",
                r"|(?-u:\b)(?:randomblob|generate_series)\s*\(|(?-u:\b)pg_sleep(?-u:\b)|(?-u:\b)lo_(?:import|export|get|put|from_bytea)\s*\(",
                r"|(?-u:\b)xp_cmdshell(?-u:\b)|(?-u:\b)select\s+(?:sleep|benchmark|pg_sleep)\s*\(",
                r"|(?-u:\b)benchmark\s*\(\s*\d",
                r"|(?-u:\b)information_schema(?-u:\b)|(?-u:\b)pg_catalog(?-u:\b)|(?-u:\b)mysql\.user(?-u:\b)|(?-u:\b)sysobjects(?-u:\b)|(?-u:\b)(?:from|join)\s+sys\.(?:objects|tables|columns)(?-u:\b)",
                r"|(?-u:\b)waitfor\s+delay\s*'",
                r"|(?-u:\b)into\s+(?:out|dump)file(?-u:\b)",
                r"|@@version(?-u:\b)",
                // Functions sqlmap and friends lean on, written as calls, as in group_concat( or char(113).
                r"|(?-u:\b)(?:group_concat|concat_ws|name_const|make_set|unhex|uncompress)\s*\(",
                r"|(?-u:\b)n?(?:char|chr)\(\s*\d",
                // version() and user() are common outside SQL, as in await version(), so only after SELECT.
                r"|(?-u:\b)(?:database|db_name|current_user|session_user|system_user|schema_name)\(\s*(?:\)|$)",
                r"|(?-u:\b)select(?:\s+\(?|\s*\()\s*(?:char|chr|cast|concat|convert|count|version|user|database|substring|ascii|if)\s*\(",
                // Conditions written as calls, as in if(1234=, which code writes with ==.
                r"|(?-u:\b)(?:if|iif|elt)\(\s*\(*\s*\d+\s*=(?:[^=]|$)|(?-u:\b)case\s+when\s+\(?\s*\d+\s*=\s*\d+\s*\)?\s+then(?-u:\b)|(?-u:\b)select\s*\(\s*case(?-u:\b)|(?-u:\b)drop\s+function\s+\w|(?-u:\b)in\s+boolean\s+mode\s*\)",
                // MySQL reads 1.e( and 1.e.tables as a number, which hides the call from filters.
                r"|\d\.e\(|\d\.e\.[a-z_]{2,}",
                // A user variable set from a subquery, as in @x:=(SELECT.
                r"|@[\w.]*\s*:?=\s*\(\s*select(?-u:\b)",
                // Database and procedure names, as in tempdb, master..sysdatabases or xp_dirtree.
                r"|(?-u:\b)(?:tempdb|msdb|sysdatabases|syscolumns|pg_shadow|pg_user|mysql\.db)(?-u:\b)|(?-u:\b)master\.(?:\.|dbo(?-u:\b))|(?-u:\b)exec\s+master\.|(?-u:\b)msysaccessobjects(?-u:\b)",
                r"|(?-u:\b)(?:xp_dirtree|xp_regread|xp_fileexist|xp_servicecontrol|xp_availablemedia|sp_password|sp_executesql|sp_oacreate|sp_makewebtask)(?-u:\b)",
                r"|(?-u:\b)(?:utl_http|utl_inaddr|dbms_pipe|dbms_lock|dbms_xmlgen|user_lock)(?-u:\b)",
                r"|(?-u:\b)declare\s+@\w|(?-u:\b)exec\s*\(\s*@|(?-u:\b)procedure\s+analyse(?-u:\b)",
                // MongoDB operators, as in u_id[$ne]=2 or {"$not": ...}. "$50 off" isn't one.
                r"|\[\$(?:ne|eq|gt|gte|lt|lte|in|nin|regex|where|exists|not|nor|or|and|expr|elemMatch|size|type|all|mod|text)\]",
                r"|^\$(?:ne|eq|gt|gte|lt|lte|in|nin|regex|where|exists|not|nor|or|and|expr|elemMatch|size|type|all|mod|text|function|accumulator)$",
            ))
            .unwrap()
        })
        .is_match(text)
}

/// SQL's SLEEP(5), but not code such as await sleep(10), time.sleep(1) or sleep(1000): SQL
/// sleeps for seconds, so a bare call with four or more digits is a delay in milliseconds.
fn sleep_call(text: &str) -> bool {
    static SLEEP: OnceLock<Regex> = OnceLock::new();
    let pattern = SLEEP.get_or_init(|| Regex::new(r"(?i)(await\s+|[.\w])?sleep\(\s*(\d+)(\s*\))?").unwrap());
    pattern.captures_iter(text).any(|c| {
        c.get(1).is_none() && !(c.get(3).is_some() && c[2].len() > 3 && text.trim().len() == c[0].len())
    })
}

/// PostgreSQL casts, as in password::int. Lower case only: Ruby and C++ names use ::Text.
fn postgres_cast(text: &str) -> bool {
    static CAST: OnceLock<Regex> = OnceLock::new();
    CAST.get_or_init(|| {
        Regex::new(r"::(?:int|integer|bigint|smallint|text|bool|boolean|jsonb?|varchar|numeric|regclass|bytea|float[48]?|double\s+precision)(?-u:\b)").unwrap()
    })
        .is_match(text)
}

/// A comment glued to a SQL keyword hides spaces from filters, as in SELECT/**/password or
/// DROP/*x*/t, and /*!50000 runs code in MySQL. Read before comments are removed.
fn comment_obfuscation(raw: &str) -> bool {
    static COMMENT: OnceLock<Regex> = OnceLock::new();
    COMMENT
        .get_or_init(|| {
            Regex::new(concat!(
                r"(?i)(?-u:\b)(?:select|union|from|where|and|or|drop|insert|update|delete|exec|order|group|having|into|values|table)/\*",
                r"|\*/(?:select|union|from|where|and|or|drop|insert|update|delete|exec|order|group|having|into|values|table)(?-u:\b)",
                r"|/\*!\d{5}|^\s*\d+\s+(?:or|and|xor)\s+/\*",
            ))
            .unwrap()
        })
        .is_match(raw)
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
            let stop = match end {
                None => length,
                Some(end) => end,
            };
            tokens.push(token(Kind::Comment, chars[index..stop].iter().collect::<String>()));
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
            // A call needs its bracket straight after the name, as in sleep(5), not prose (like this).
            let kind = if KEYWORDS.contains(&word.as_str()) {
                Kind::Keyword
            } else if end < length && chars[end] == '(' {
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
    let mut depth = 0i32;
    for token in tokens {
        if token.kind == Kind::Comment {
            break;
        }
        match token.kind {
            Kind::Open => depth += 1,
            Kind::Close => depth -= 1,
            _ => {}
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
        if atom && previous_atom && depth <= 0 {
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

/// After a bare word, require a UNION SELECT, a comparison, a function call or a statement.
fn word_breaks_out(tokens: &[Token]) -> bool {
    let mut index = 0;
    while index < tokens.len() && tokens[index].kind == Kind::Close {
        index += 1;
    }
    let Some(first) = tokens.get(index) else {
        return false;
    };
    let rest = &tokens[index + 1..];
    if first.kind == Kind::Semicolon {
        return rest.first().is_some_and(|t| is_keyword(t, STATEMENTS)) && statement(rest);
    }
    if first.kind != Kind::Keyword || !CONNECTORS.contains(&first.text.as_str()) {
        return false;
    }
    if has_union_select(&tokens[index..]) {
        return true;
    }
    first.text != "union"
        && valid(rest)
        && rest.iter().any(|t| {
            matches!(t.kind, Kind::Function | Kind::Semicolon)
                || (t.kind == Kind::Operator && COMPARISONS.contains(&t.text.as_str()))
                || is_keyword(t, &["select"])
        })
}

fn ends_with_hiding_comment(tokens: &[Token]) -> bool {
    tokens.iter().find(|t| t.kind == Kind::Comment).is_some_and(|t| comment_ends_value(&t.text))
}

/// A value that starts with OR, AND or XOR and then a comparison, as in AND 2771=1709) or
/// or 'a' <. The query's own value came before it. Ordinary text such as "or more" has no
/// comparison.
fn starts_with_connector(tokens: &[Token]) -> bool {
    let Some(first) = tokens.first() else {
        return false;
    };
    if !is_keyword(first, &["or", "and", "xor", "having"]) || tokens.len() < 3 {
        return false;
    }
    let rest = &tokens[1..];
    let opens_with_value = matches!(rest[0].kind, Kind::Number | Kind::Str | Kind::Word | Kind::Open | Kind::Function);
    let compares = rest.iter().any(|t| {
        (t.kind == Kind::Operator && COMPARISONS.contains(&t.text.as_str())) || t.kind == Kind::Function || is_keyword(t, &["select"])
    });
    opens_with_value && compares && !rest.iter().any(|t| t.kind == Kind::Unknown)
}

/// A statement and its object, as in INSERT INTO, DELETE FROM, DROP TABLE or UPDATE t SET.
fn statement_start(tokens: &[Token]) -> bool {
    match (tokens.first(), tokens.get(1)) {
        (Some(a), Some(b)) => {
            (is_keyword(a, &["insert"]) && is_keyword(b, &["into"]))
                || (is_keyword(a, &["delete"]) && is_keyword(b, &["from"]))
                || (is_keyword(a, &["drop", "alter", "truncate", "create"]) && is_keyword(b, &["table"]))
                || (is_keyword(a, &["update"]) && b.kind == Kind::Word && tokens.get(2).is_some_and(|t| is_keyword(t, &["set"])))
        }
        _ => false,
    }
}

/// SQL that starts the value, as in in (select * from, union select col from, except values(
/// or ALTER TABLE t ADD. Each needs more than a phrase such as "union select committee".
fn statement_shape(text: &str) -> bool {
    static SHAPE: OnceLock<Regex> = OnceLock::new();
    SHAPE
        .get_or_init(|| {
            Regex::new(concat!(
                r"(?i)^\s*(?:not\s+)?in\s*\(\s*select(?-u:\b).*(?-u:\b)from(?-u:\b)",
                r"|^\s*union\s+(?:all\s+|distinct\s+)?select\s+\S.*(?-u:\b)from(?-u:\b)",
                r"|^\s*(?:except|intersect)\s+(?:select[\s.(]+(?:\d|\*|null(?-u:\b))|values\s*\()",
                r"|^\s*alter\s+table\s+\S+\s+(?:add|change|drop|modify|rename)(?-u:\b)",
                r"|^\s*create\s+table\s+\S+\s*\(",
            ))
            .unwrap()
        })
        .is_match(text)
}

/// A whole statement as the value, as in SELECT * FROM users, ; insert into or DROP t;--.
fn statement_value(tokens: &[Token]) -> bool {
    let Some(first) = tokens.first() else {
        return false;
    };
    // After a semicolon, a statement and its object are enough, as in ; insert into.
    if first.kind == Kind::Semicolon {
        return statement_start(&tokens[1..]);
    }
    if is_keyword(first, &["select"]) {
        // SELECT ... FROM needs more than prose has, as in * or a subquery: not "select size from chart".
        let from = tokens.iter().position(|t| is_keyword(t, &["from"]));
        return from.is_some_and(|i| i > 1)
            && tokens.iter().any(|t| {
                matches!(t.kind, Kind::Open | Kind::Function | Kind::Comment)
                    || (t.kind == Kind::Operator && (t.text == "*" || COMPARISONS.contains(&t.text.as_str())))
                    || is_keyword(t, &["union"])
            })
            && !tokens.iter().any(|t| matches!(t.kind, Kind::Unknown | Kind::Unterminated));
    }
    // DROP TABLE t;-- and friends: a statement that ends the query and hides the rest. Code
    // such as delete Math.pow; isn't one.
    if statement_start(tokens) || is_keyword(first, &["exec", "execute"]) {
        let semicolon = tokens.iter().position(|t| t.kind == Kind::Semicolon);
        return semicolon.is_some_and(|i| {
            i > 1 && valid(&tokens[..i]) && tokens[i + 1..].iter().all(|t| t.kind == Kind::Comment)
        });
    }
    // DROP t;-- names no TABLE, so the comment after it is needed, as code has none.
    if is_keyword(first, &["drop", "truncate"]) && tokens.len() == 4 {
        return tokens[1].kind == Kind::Word
            && tokens[2].kind == Kind::Semicolon
            && tokens[3].kind == Kind::Comment
            && comment_ends_value(&tokens[3].text);
    }
    false
}

/// True for a comment that hides the rest of the query, as in admin'-- or admin'#, and not for
/// text such as '--INFANT or 7" #2, where MySQL needs a space after -- and # starts more text.
fn comment_ends_value(text: &str) -> bool {
    // The tail ends with the query's own closing quote, which isn't part of the value.
    let text = text.strip_suffix(['\'', '"']).unwrap_or(text);
    if let Some(after) = text.strip_prefix("--") {
        return after.chars().next().is_none_or(|c| c.is_whitespace() || c == '-' || c == '+');
    }
    if let Some(after) = text.strip_prefix('#') {
        return after.trim().is_empty();
    }
    true
}

fn strong(tokens: &[Token]) -> bool {
    tokens.iter().any(|t| {
        matches!(t.kind, Kind::Function | Kind::Comment | Kind::Semicolon)
            || (t.kind == Kind::Operator && COMPARISONS.contains(&t.text.as_str()))
            || is_keyword(t, STRONG_KEYWORDS)
    })
}

/// True when the tokens contain UNION [ALL|DISTINCT] [(] SELECT followed by something.
fn has_union_select(tokens: &[Token]) -> bool {
    tokens.iter().enumerate().any(|(i, t)| {
        is_keyword(t, &["union"]) && {
            let mut rest = tokens[i + 1..]
                .iter()
                .skip_while(|t| t.kind == Kind::Open || t.text == "all" || t.text == "distinct");
            rest.next().is_some_and(|t| is_keyword(t, &["select"])) && rest.next().is_some_and(|t| t.kind != Kind::Comment)
        }
    })
}

/// A statement keyword after the value ends must start a plausible statement, as in
/// ; DROP TABLE users, and not text such as ; exec will create the case.
fn statement(tokens: &[Token]) -> bool {
    let Some(first) = tokens.first() else {
        return false;
    };
    if is_keyword(first, &["set"]) {
        // SET needs an assignment, as in ; SET @a = 1, not text such as BASE; SET SWAROVSKI.
        return tokens.iter().any(|t| t.kind == Kind::Operator && (t.text == "=" || t.text == ":=")) && valid(tokens);
    }
    if is_keyword(first, &["exec", "execute"]) {
        let name_ok = tokens.get(1).is_some_and(|t| matches!(t.kind, Kind::Word | Kind::Function));
        let then_ok = tokens.get(2).is_none_or(|t| t.kind != Kind::Keyword);
        return name_ok && then_ok && valid(tokens);
    }
    valid(tokens)
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
        Kind::Comment => return !numeric && comment_ok && comment_ends_value(&first.text),
        Kind::Semicolon => return rest.first().is_some_and(|t| is_keyword(t, STATEMENTS)) && statement(rest),
        Kind::Keyword if BARE_STATEMENTS.contains(&first.text.as_str()) => return statement(&tokens[index..]),
        Kind::Keyword if CONNECTORS.contains(&first.text.as_str()) => {}
        // A column alias and FROM, as in ) AS orders FROM. After a number the alias must be
        // quoted, so "5 as gift from" stays text. After a bracket, as in ) AS t FROM, it needn't be.
        Kind::Keyword if first.text == "as" => {
            return rest.get(1).is_some_and(|t| is_keyword(t, &["from"]))
                && rest.first().is_some_and(|t| match t.kind {
                    Kind::Str => true,
                    Kind::Word => !numeric || index > 0,
                    _ => false,
                });
        }
        // NOT LIKE, NOT IN and so on, as in id'not like id--.
        Kind::Keyword if first.text == "not" && rest.first().is_some_and(|t| is_keyword(t, CONNECTORS)) => {
            return breaks_out(rest, numeric, comment_ok);
        }
        Kind::Operator if !numeric && OPERATOR_CONNECTORS.contains(&first.text.as_str()) => {}
        // A binary operator and a value count when a comment then hides the rest, as in
        // x'|email--. Not ~ or !, which take no left side, and not ASCII art such as '-----.
        Kind::Operator
            if !numeric
                && comment_ok
                && !matches!(first.text.as_str(), "~" | "!")
                && rest.first().is_some_and(|t| matches!(t.kind, Kind::Word | Kind::Number | Kind::Str | Kind::Function | Kind::Open))
                && ends_with_hiding_comment(rest)
                && valid(rest) =>
        {
            return true
        }
        _ => return false,
    }
    if has_union_select(rest) {
        return true;
    }
    if first.text == "union" {
        let next = rest
            .iter()
            .find(|t| t.kind != Kind::Open && t.text != "all" && t.text != "distinct");
        if !next.is_some_and(|t| is_keyword(t, &["select"])) {
            return false;
        }
        // UNION SELECT is decisive, whatever its column list holds, as in from table/*.
        return rest.iter().filter(|t| t.kind != Kind::Comment).count() >= 2;
    }
    if (first.text == "order" || first.text == "group") && !rest.first().is_some_and(|t| is_keyword(t, &["by"])) {
        return false;
    }
    // ORDER BY and a column number counts columns for a UNION, as in -8970) ORDER BY 3631.
    if first.text == "order" && rest.get(1).is_some_and(|t| t.kind == Kind::Number) && valid(rest) {
        return true;
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
    // + is a space in form-encoded text, as in Use+'in'+or+'between', and Markdown puts code
    // in backticks, as in a `'limit'` event.
    if end > 0 && !" +([{,:=`".contains(chars[end - 1]) {
        return false;
    }
    let mut stop = end + 1;
    while stop < chars.len() && chars[stop].is_alphabetic() {
        stop += 1;
    }
    // The word closes and text follows, as in "IS" STAMP, not more SQL, as in 'or'1'='1.
    let followed_by_text = chars.get(stop + 1).is_none_or(|c| c.is_whitespace() || "),.;:!?]}+`".contains(*c));
    stop < chars.len()
        && chars[stop] == quote
        && followed_by_text
        && KEYWORDS.contains(&chars[end + 1..stop].iter().collect::<String>().to_lowercase().as_str())
}

/// Returns a rule name when the value breaks out of a SQL value, otherwise `None`.
pub fn classify(value: &str) -> Option<&'static str> {
    let cleaned = clean(value);
    let chars: Vec<char> = cleaned.chars().collect();
    let tokens = lex(&chars);
    // A value that starts by closing a call or statement, as in ) UNION SELECT or ; DROP TABLE.
    if tokens.first().is_some_and(|t| matches!(t.kind, Kind::Close | Kind::Semicolon)) && breaks_out(&tokens, true, true) {
        return Some("sql_breakout_numeric");
    }
    let mut start = usize::from(
        tokens.len() > 1 && tokens[0].kind == Kind::Operator && (tokens[0].text == "-" || tokens[0].text == "+"),
    );
    if tokens.get(start).is_some_and(|t| t.kind == Kind::Number) {
        // The number can be an expression, as in -6863 union or 189-999.9 union.
        while tokens.get(start + 1).is_some_and(|t| t.kind == Kind::Operator && "+-*/%".contains(t.text.as_str()))
            && tokens.get(start + 2).is_some_and(|t| t.kind == Kind::Number)
        {
            start += 2;
        }
        if breaks_out(&tokens[start + 1..], true, true) {
            return Some("sql_breakout_numeric");
        }
        let after = &tokens[start + 1..];
        if after.first().is_some_and(|t| t.kind == Kind::Operator && "+-*/%".contains(t.text.as_str()))
            && after.get(1).is_some_and(|t| t.kind == Kind::Open)
            && after.get(2).is_some_and(|t| is_keyword(t, &["select"]))
            && valid(&after[1..])
        {
            return Some("sql_breakout_numeric");
        }
    }
    if starts_with_connector(&tokens) || statement_value(&tokens) || statement_shape(&cleaned) {
        return Some("sql_breakout_numeric");
    }
    // A bare word can be the value too, as in foo) UNION ALL SELECT or foo WHERE 1=1 UNION ...
    // Only decisive evidence counts here, so "price between 10 and 20" stays ordinary text.
    if tokens.first().is_some_and(|t| t.kind == Kind::Word) && word_breaks_out(&tokens[1..]) {
        return Some("sql_breakout_numeric");
    }
    // A word plus a subquery, as in foo + (SELECT 1 WHERE 2=2). The subquery needs FROM, WHERE
    // or a comparison, so "cheese + (select a topping)" stays text.
    if tokens.first().is_some_and(|t| t.kind == Kind::Word)
        && tokens.get(1).is_some_and(|t| t.kind == Kind::Operator && matches!(t.text.as_str(), "+" | "-" | "||"))
        && tokens.get(2).is_some_and(|t| t.kind == Kind::Open)
        && tokens.get(3).is_some_and(|t| is_keyword(t, &["select"]))
        && valid(&tokens[2..])
        && tokens[3..].iter().any(|t| {
            is_keyword(t, &["from", "where"]) || (t.kind == Kind::Operator && COMPARISONS.contains(&t.text.as_str()))
        })
    {
        return Some("sql_breakout_numeric");
    }
    for (quote, rule) in [('\'', "sql_breakout_single_quote"), ('"', "sql_breakout_double_quote")] {
        if !chars.contains(&quote) {
            continue;
        }
        let odd = chars.iter().filter(|&&c| c == quote).count() % 2 == 1;
        let first = chars.iter().position(|c| !c.is_whitespace());
        if odd && first.is_some_and(|i| chars[i] == quote) {
            // The value is a quoted string compared with the query's own, as in 'x'!='y.
            let mut whole: Vec<char> = chars.clone();
            whole.push(quote);
            let tokens = lex(&whole);
            if tokens.first().is_some_and(|t| t.kind == Kind::Str) && breaks_out(&tokens[1..], false, true) {
                return Some(rule);
            }
            // UNION or ALTER straight after the opening quote, as in 'UNION A or ' alter/.
            let after: String = chars[first.unwrap_or(0) + 1..].iter().collect::<String>().to_ascii_lowercase();
            let after = after.trim_start();
            if ["union", "alter"].iter().any(|w| {
                after.strip_prefix(w).is_some_and(|r| r.starts_with(|c: char| c.is_whitespace() || c == '/'))
            }) {
                return Some(rule);
            }
        }
        // MySQL treats a backslash as an escape and most other databases don't, so try both.
        let modes: &[bool] = if chars.contains(&'\\') { &[true, false] } else { &[false] };
        for &backslash in modes {
            if let Some(end) = closing_quote(&chars, quote, backslash).filter(|&end| !quoted_word(&chars, end, quote)) {
                // The query's own closing quote follows the value.
                let mut tail: Vec<char> = chars[end + 1..].to_vec();
                tail.push(quote);
                let odd = chars.iter().filter(|&&c| c == quote).count() % 2 == 1;
                let tail_tokens = lex(&tail);
                if breaks_out(&tail_tokens, false, odd) {
                    return Some(rule);
                }
                // Filters are dodged by dropping spaces, as in a'or'1'='1. Prose has a space after
                // the quote, as in "parents' or yours", so a glued connector is evidence enough.
                let glued = chars.get(end + 1).is_some_and(|c| c.is_alphabetic());
                if glued
                    && tail_tokens.first().is_some_and(|t| is_keyword(t, &["or", "and", "xor", "like", "rlike", "regexp"]))
                    && valid(&tail_tokens[1..])
                    && !tail_tokens.iter().any(|t| t.kind == Kind::Unterminated)
                    && tail_tokens.len() > 2
                {
                    return Some(rule);
                }
                // The value ends the statement, as in admin'; or admin'/* x */;. Prose such as
                // "John's;" has a letter after the quote, and 18'; is a size in feet.
                if odd && !(end > 0 && chars[end - 1].is_ascii_digit()) {
                    let rest = lex(&chars[end + 1..]);
                    if rest.iter().any(|t| t.kind == Kind::Semicolon)
                        && rest.iter().all(|t| matches!(t.kind, Kind::Semicolon | Kind::Comment))
                    {
                        return Some(rule);
                    }
                }
                // Blind injection often leaves the query's quote unmatched, as in ' and 526=527.
                // Databases reject that, but CRS and libinjection treat it as an attack. Only a
                // keyword with more evidence after it counts, so 15"+LAPTOP+CASE stays text.
                if odd && breaks_out(&lex(&chars[end + 1..]), true, false) {
                    return Some(rule);
                }
            }
        }
    }
    // Names that only appear in SQL, as in pg_sleep( or information_schema.
    if sql_only_name(&cleaned) || postgres_cast(&cleaned) || sleep_call(&cleaned) {
        return Some("sql_function");
    }
    if comment_obfuscation(value) {
        return Some("sql_comment");
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
