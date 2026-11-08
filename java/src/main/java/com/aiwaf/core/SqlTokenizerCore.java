package com.aiwaf.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Finds values that break out of a SQL value and continue as SQL.
 *
 * <p>Each value is tried as a bare number, inside single quotes, and inside double quotes. It's
 * flagged only when, in one of them, it closes the value and what follows reads as a SQL condition
 * or statement. The main implementation is Rust ({@code aiwaf_core::sql}); this copy is for Java,
 * which doesn't use Rust. The shared cases in {@code tests/fixtures/sql_injection_cases.json} run
 * against every copy.
 */
public final class SqlTokenizerCore {
    private static final Set<String> KEYWORDS = words("all alter and as asc between by case collate create declare delay delete desc distinct div drop else end exec execute exists false from group having in insert into is join like limit not null offset on or order outfile procedure regexp rlike select set shutdown sounds table then true truncate union update values waitfor when where xor");
    private static final Set<String> CONNECTORS = words("and between collate div group having in into is like limit or order procedure regexp rlike sounds union where xor");
    // MSSQL runs a second statement without a semicolon, as in ' WAITFOR DELAY '0:0:5'.
    private static final Set<String> BARE_STATEMENTS = words("declare exec execute waitfor");
    // < and > are left out: quoted HTML such as ('<script>') would read as a comparison.
    private static final Set<String> OPERATOR_CONNECTORS = words("= <> != <=> || && +");
    private static final Set<String> COMPARISONS = words("= <> != < > <= >= <=>");
    private static final Set<String> STATEMENTS = words("alter create declare delete drop exec execute insert select set shutdown truncate update waitfor");
    private static final Set<String> STRONG_KEYWORDS = words("between exists group having in into is like limit order procedure regexp rlike select");
    private static final Set<String> VALUE_KEYWORDS = words("false null true");
    private static final Set<String> END_KEYWORDS = words("asc desc end false null true");
    private static final List<String> MULTI_OPERATORS = List.of("<=>", "<>", "!=", "<=", ">=", "||", "&&", "::", ":=");
    private static final String SINGLE_OPERATORS = "=<>+-*/%&|^~!";
    // A null byte counts as a space: sqlmap ends payloads with one, as in -8970) ORDER BY 3631%00.
    private static final Set<Integer> WHITESPACE = Set.of(0x00, 0x20, 0x09, 0x0a, 0x0d, 0x0b, 0x0c, 0xa0, 0x1680, 0x2000, 0x2001,
            0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000, 0xfeff);
    private static final Pattern VERSIONED_COMMENT = Pattern.compile("/\\*!\\d{0,6}([\\s\\S]*?)\\*/");
    private static final Pattern COMMENT = Pattern.compile("/\\*[\\s\\S]*?\\*/");
    // Names that only appear in SQL, as in pg_sleep( or information_schema.
    private static final Pattern SQL_ONLY_NAME = Pattern.compile(
            "(?i)\\b(?:pg_sleep|load_file|extractvalue|updatexml|dbms_pipe\\.\\w+|utl_inaddr\\.\\w+|utl_http\\.\\w+|sys_eval|sys_exec)\\s*\\("
            + "|\\b(?:randomblob|generate_series)\\s*\\(|\\bpg_sleep\\b|\\blo_(?:import|export|get|put|from_bytea)\\s*\\("
            + "|\\bxp_cmdshell\\b|\\bselect\\s+(?:sleep|benchmark|pg_sleep)\\s*\\("
            + "|\\bbenchmark\\s*\\(\\s*\\d"
            + "|\\binformation_schema\\b|\\bpg_catalog\\b|\\bmysql\\.user\\b|\\bsysobjects\\b|\\b(?:from|join)\\s+sys\\.(?:objects|tables|columns)\\b"
            + "|\\bwaitfor\\s+delay\\s*'"
            + "|\\binto\\s+(?:out|dump)file\\b"
            + "|@@version\\b"
            // Functions sqlmap and friends lean on, written as calls, as in group_concat( or char(113).
            + "|\\b(?:group_concat|concat_ws|name_const|make_set|unhex|uncompress)\\s*\\("
            + "|\\bn?(?:char|chr)\\(\\s*\\d"
            // version() and user() are common outside SQL, as in await version(), so only after SELECT.
            + "|\\b(?:database|db_name|current_user|session_user|system_user|schema_name)\\(\\s*(?:\\)|$)"
            + "|\\bselect(?:\\s+\\(?|\\s*\\()\\s*(?:char|chr|cast|concat|convert|count|version|user|database|substring|ascii|if)\\s*\\("
            // Conditions written as calls, as in if(1234=, which code writes with ==.
            + "|\\b(?:if|iif|elt)\\(\\s*\\(*\\s*\\d+\\s*=(?:[^=]|$)|\\bcase\\s+when\\s+\\(?\\s*\\d+\\s*=\\s*\\d+\\s*\\)?\\s+then\\b"
            + "|\\bselect\\s*\\(\\s*case\\b|\\bdrop\\s+function\\s+\\w|\\bin\\s+boolean\\s+mode\\s*\\)"
            // MySQL reads 1.e( and 1.e.tables as a number, which hides the call from filters.
            + "|\\d\\.e\\(|\\d\\.e\\.[a-z_]{2,}"
            // A user variable set from a subquery, as in @x:=(SELECT.
            + "|@[\\w.]*\\s*:?=\\s*\\(\\s*select\\b"
            // Database and procedure names, as in tempdb, master..sysdatabases or xp_dirtree.
            + "|\\b(?:tempdb|msdb|sysdatabases|syscolumns|pg_shadow|pg_user|mysql\\.db)\\b|\\bmaster\\.(?:\\.|dbo\\b)|\\bexec\\s+master\\.|\\bmsysaccessobjects\\b"
            + "|\\b(?:xp_dirtree|xp_regread|xp_fileexist|xp_servicecontrol|xp_availablemedia|sp_password|sp_executesql|sp_oacreate|sp_makewebtask)\\b"
            + "|\\b(?:utl_http|utl_inaddr|dbms_pipe|dbms_lock|dbms_xmlgen|user_lock)\\b"
            + "|\\bdeclare\\s+@\\w|\\bexec\\s*\\(\\s*@|\\bprocedure\\s+analyse\\b"
            // MongoDB operators, as in u_id[$ne]=2 or {"$not": ...}. "$50 off" isn't one.
            + "|\\[\\$(?:ne|eq|gt|gte|lt|lte|in|nin|regex|where|exists|not|nor|or|and|expr|elemMatch|size|type|all|mod|text)\\]"
            + "|^\\$(?:ne|eq|gt|gte|lt|lte|in|nin|regex|where|exists|not|nor|or|and|expr|elemMatch|size|type|all|mod|text|function|accumulator)$");
    // PostgreSQL casts, as in password::int. Lower case only: Ruby and C++ names use ::Text.
    private static final Pattern POSTGRES_CAST =
            Pattern.compile("::(?:int|integer|bigint|smallint|text|bool|boolean|jsonb?|varchar|numeric|regclass|bytea|float[48]?|double\\s+precision)\\b");
    // A comment glued to a SQL keyword hides spaces from filters, as in SELECT/**/password or
    // DROP/*x*/t, and /*!50000 runs code in MySQL. Read before comments are removed.
    private static final Pattern COMMENT_OBFUSCATION = Pattern.compile(
            "(?i)\\b(?:select|union|from|where|and|or|drop|insert|update|delete|exec|order|group|having|into|values|table)/\\*"
            + "|\\*/(?:select|union|from|where|and|or|drop|insert|update|delete|exec|order|group|having|into|values|table)\\b"
            + "|/\\*!\\d{5}|^\\s*\\d+\\s+(?:or|and|xor)\\s+/\\*");
    // SQL that starts the value, as in in (select * from, union select col from, except values(
    // or ALTER TABLE t ADD. Each needs more than a phrase such as "union select committee".
    private static final Pattern STATEMENT_SHAPE = Pattern.compile(
            "(?i)^\\s*(?:not\\s+)?in\\s*\\(\\s*select\\b.*\\bfrom\\b"
            + "|^\\s*union\\s+(?:all\\s+|distinct\\s+)?select\\s+\\S.*\\bfrom\\b"
            + "|^\\s*(?:except|intersect)\\s+(?:select[\\s.(]+(?:\\d|\\*|null\\b)|values\\s*\\()"
            + "|^\\s*alter\\s+table\\s+\\S+\\s+(?:add|change|drop|modify|rename)\\b"
            + "|^\\s*create\\s+table\\s+\\S+\\s*\\(");
    // SQL's SLEEP(5), but not code such as await sleep(10), time.sleep(1) or sleep(1000): SQL
    // sleeps for seconds, so a bare call with four or more digits is a delay in milliseconds.
    private static final Pattern SLEEP_CALL = Pattern.compile("(?i)(await\\s+|[.\\w])?sleep\\(\\s*(\\d+)(\\s*\\))?");
    private static final Set<String> GLUED_CONNECTORS = words("or and xor like rlike regexp");

    private SqlTokenizerCore() {}

    record Token(char kind, String text) {}

    private static Set<String> words(String text) {
        return Set.of(text.split(" "));
    }

    /** Keeps MySQL's executed comments, drops other block comments, and unifies whitespace. */
    public static String clean(String value) {
        String text = VERSIONED_COMMENT.matcher(value == null ? "" : value).replaceAll(" $1 ");
        text = COMMENT.matcher(text).replaceAll(" ");
        StringBuilder out = new StringBuilder(text.length());
        // Apps often normalise typographic quotes to ', as in %u2019 OR 1=1-- -.
        text.codePoints().forEach(c -> out.appendCodePoint(WHITESPACE.contains(c) ? ' '
                : (c == 0x2018 || c == 0x2019 || c == 0x02bc || c == 0xff07) ? '\'' : c));
        return out.toString();
    }

    private static boolean digit(int c) {
        return c >= '0' && c <= '9';
    }

    private static boolean letter(int c) {
        return Character.isAlphabetic(c);
    }

    private static boolean wordChar(int c) {
        return letter(c) || digit(c) || "_$@.".indexOf(c) >= 0;
    }

    private static boolean startsWith(int[] chars, int index, String part) {
        int[] wanted = part.codePoints().toArray();
        if (index + wanted.length > chars.length) return false;
        for (int i = 0; i < wanted.length; i++) {
            if (chars[index + i] != wanted[i]) return false;
        }
        return true;
    }

    private static int find(int[] chars, int from, String part) {
        for (int i = from; i < chars.length; i++) {
            if (startsWith(chars, i, part)) return i;
        }
        return -1;
    }

    private static String text(int[] chars, int start, int end) {
        return new String(chars, start, end - start);
    }

    static List<Token> lex(String value) {
        int[] chars = value.codePoints().toArray();
        int length = chars.length;
        List<Token> tokens = new ArrayList<>();
        int index = 0;
        while (index < length) {
            int c = chars[index];
            if (c == ' ') {
                index++;
            } else if (c == '\'' || c == '"' || c == '`') {
                int end = index + 1;
                while (end < length) {
                    if (chars[end] == '\\' && c != '`') end += 2;
                    else if (chars[end] == c && end + 1 < length && chars[end + 1] == c) end += 2;
                    else if (chars[end] == c) break;
                    else end++;
                }
                if (end >= length) {
                    tokens.add(new Token('u', ""));
                    return tokens;
                }
                tokens.add(new Token(c == '`' ? 'v' : 's', text(chars, index + 1, end)));
                index = end + 1;
            } else if (c == '#' || startsWith(chars, index, "--") || startsWith(chars, index, "/*")) {
                boolean block = c == '/';
                int end = block ? find(chars, index + 2, "*/") : find(chars, index, "\n");
                tokens.add(new Token('c', text(chars, index, end < 0 ? length : end)));
                index = end < 0 ? length : end + (block ? 2 : 1);
            } else if (digit(c) || (c == '.' && index + 1 < length && digit(chars[index + 1]))) {
                int end = index + 1;
                if (c == '0' && end < length && (chars[end] == 'x' || chars[end] == 'X')) {
                    end++;
                    while (end < length && Character.digit(chars[end], 16) >= 0 && chars[end] < 128) end++;
                } else {
                    while (end < length && (digit(chars[end]) || chars[end] == '.')) end++;
                    if (end + 1 < length && (chars[end] == 'e' || chars[end] == 'E')
                            && (digit(chars[end + 1]) || chars[end + 1] == '+' || chars[end + 1] == '-')) {
                        end += 2;
                        while (end < length && digit(chars[end])) end++;
                    }
                }
                if (end < length && (letter(chars[end]) || chars[end] == '_')) {
                    while (end < length && wordChar(chars[end])) end++;
                    tokens.add(new Token('v', text(chars, index, end).toLowerCase(java.util.Locale.ROOT)));
                } else {
                    tokens.add(new Token('n', text(chars, index, end)));
                }
                index = end;
            } else if (letter(c) || "_$@".indexOf(c) >= 0) {
                int end = index + 1;
                while (end < length && wordChar(chars[end])) end++;
                String word = text(chars, index, end).toLowerCase(java.util.Locale.ROOT);
                // A call needs its bracket straight after the name, as in sleep(5), not prose (like this).
                int after = end;
                if (KEYWORDS.contains(word)) tokens.add(new Token('k', word));
                else if (after < length && chars[after] == '(') tokens.add(new Token('f', word));
                else tokens.add(new Token('v', word));
                index = end;
            } else if ("(),;".indexOf(c) >= 0) {
                tokens.add(new Token((char) c, String.valueOf((char) c)));
                index++;
            } else {
                String operator = null;
                for (String item : MULTI_OPERATORS) {
                    if (startsWith(chars, index, item)) {
                        operator = item;
                        break;
                    }
                }
                if (operator == null && SINGLE_OPERATORS.indexOf(c) >= 0) operator = String.valueOf((char) c);
                if (operator == null) {
                    tokens.add(new Token('x', new String(Character.toChars(c))));
                    index++;
                } else {
                    tokens.add(new Token('o', operator));
                    index += operator.length();
                }
            }
        }
        return tokens;
    }

    private static boolean isKeyword(Token token, Set<String> words) {
        return token.kind() == 'k' && words.contains(token.text());
    }

    private static boolean isAtom(Token token) {
        return "nsv".indexOf(token.kind()) >= 0 || isKeyword(token, VALUE_KEYWORDS);
    }

    /** Rejects sequences no SQL dialect accepts: adjacent values, open strings, dangling ends. */
    private static boolean valid(List<Token> tokens) {
        Token last = null;
        boolean previousAtom = false;
        int depth = 0;
        for (Token token : tokens) {
            if (token.kind() == 'c') break;
            if (token.kind() == '(') depth++;
            if (token.kind() == ')') depth--;
            if (token.kind() == 'x' || token.kind() == 'u') return false;
            if (last != null && last.kind() == 'f' && token.kind() != '(') return false;
            boolean atom = isAtom(token);
            if (atom && previousAtom && depth <= 0) return false;
            previousAtom = atom || token.kind() == ')';
            last = token;
        }
        if (last == null || "o(,f;".indexOf(last.kind()) >= 0) return false;
        return !(last.kind() == 'k' && !END_KEYWORDS.contains(last.text()));
    }

    private static boolean strong(List<Token> tokens) {
        for (Token token : tokens) {
            if ("fc;".indexOf(token.kind()) >= 0
                    || (token.kind() == 'o' && COMPARISONS.contains(token.text()))
                    || isKeyword(token, STRONG_KEYWORDS)) {
                return true;
            }
        }
        return false;
    }

    /** After a bare word, require a UNION SELECT, a comparison, a function call or a statement. */
    private static boolean wordBreaksOut(List<Token> tokens) {
        int index = 0;
        while (index < tokens.size() && tokens.get(index).kind() == ')') index++;
        if (index >= tokens.size()) return false;
        Token first = tokens.get(index);
        List<Token> rest = tokens.subList(index + 1, tokens.size());
        if (first.kind() == ';') return !rest.isEmpty() && isKeyword(rest.get(0), STATEMENTS) && statement(rest);
        if (!isKeyword(first, CONNECTORS)) return false;
        if (hasUnionSelect(tokens.subList(index, tokens.size()))) return true;
        if (first.text().equals("union") || !valid(rest)) return false;
        for (Token token : rest) {
            if (token.kind() == 'f' || token.kind() == ';' || (token.kind() == 'o' && COMPARISONS.contains(token.text()))
                    || isKeyword(token, Set.of("select"))) {
                return true;
            }
        }
        return false;
    }

    /**
     * True for a comment that hides the rest of the query, as in admin'-- or admin'#, and not for
     * text such as '--INFANT or 7" #2, where MySQL needs a space after -- and # starts more text.
     */
    private static boolean commentEndsValue(String text) {
        // The tail ends with the query's own closing quote, which isn't part of the value.
        String value = text.endsWith("'") || text.endsWith("\"") ? text.substring(0, text.length() - 1) : text;
        if (value.startsWith("--")) {
            String after = value.substring(2);
            return after.isEmpty() || Character.isWhitespace(after.charAt(0)) || after.charAt(0) == '-' || after.charAt(0) == '+';
        }
        if (value.startsWith("#")) return value.substring(1).trim().isEmpty();
        return true;
    }

    private static boolean endsWithHidingComment(List<Token> tokens) {
        return tokens.stream().filter(t -> t.kind() == 'c').findFirst().map(t -> commentEndsValue(t.text())).orElse(false);
    }

    /**
     * A value that starts with OR, AND or XOR and then a comparison, as in AND 2771=1709) or
     * or 'a' <. The query's own value came before it. Ordinary text such as "or more" has no
     * comparison.
     */
    private static boolean startsWithConnector(List<Token> tokens) {
        if (tokens.size() < 3 || !isKeyword(tokens.get(0), Set.of("or", "and", "xor", "having"))) return false;
        List<Token> rest = tokens.subList(1, tokens.size());
        boolean opensWithValue = "nsv(f".indexOf(rest.get(0).kind()) >= 0;
        boolean compares = rest.stream().anyMatch(t -> (t.kind() == 'o' && COMPARISONS.contains(t.text()))
                || t.kind() == 'f' || isKeyword(t, Set.of("select")));
        return opensWithValue && compares && rest.stream().noneMatch(t -> t.kind() == 'x');
    }

    /** A statement and its object, as in INSERT INTO, DELETE FROM, DROP TABLE or UPDATE t SET. */
    private static boolean statementStart(List<Token> tokens) {
        if (tokens.size() < 2) return false;
        Token a = tokens.get(0);
        Token b = tokens.get(1);
        return (isKeyword(a, Set.of("insert")) && isKeyword(b, Set.of("into")))
                || (isKeyword(a, Set.of("delete")) && isKeyword(b, Set.of("from")))
                || (isKeyword(a, Set.of("drop", "alter", "truncate", "create")) && isKeyword(b, Set.of("table")))
                || (isKeyword(a, Set.of("update")) && b.kind() == 'v' && tokens.size() > 2 && isKeyword(tokens.get(2), Set.of("set")));
    }

    /** A whole statement as the value, as in SELECT * FROM users, ; insert into or DROP TABLE t;--. */
    private static boolean statementValue(List<Token> tokens) {
        if (tokens.isEmpty()) return false;
        Token first = tokens.get(0);
        // After a semicolon, a statement and its object are enough, as in ; insert into.
        if (first.kind() == ';') return statementStart(tokens.subList(1, tokens.size()));
        if (isKeyword(first, Set.of("select"))) {
            // SELECT ... FROM needs more than prose has, as in * or a subquery: not "select size from chart".
            int from = -1;
            for (int i = 0; i < tokens.size(); i++) {
                if (isKeyword(tokens.get(i), Set.of("from"))) {
                    from = i;
                    break;
                }
            }
            return from > 1
                    && tokens.stream().anyMatch(t -> "(fc".indexOf(t.kind()) >= 0
                            || (t.kind() == 'o' && (t.text().equals("*") || COMPARISONS.contains(t.text())))
                            || isKeyword(t, Set.of("union")))
                    && tokens.stream().noneMatch(t -> t.kind() == 'x' || t.kind() == 'u');
        }
        // DROP TABLE t;-- and friends: a statement that ends the query and hides the rest. Code
        // such as delete Math.pow; isn't one.
        if (statementStart(tokens) || isKeyword(first, Set.of("exec", "execute"))) {
            for (int i = 0; i < tokens.size(); i++) {
                if (tokens.get(i).kind() != ';') continue;
                if (i <= 1 || !valid(tokens.subList(0, i))) return false;
                return tokens.subList(i + 1, tokens.size()).stream().allMatch(t -> t.kind() == 'c');
            }
            return false;
        }
        // DROP t;-- names no TABLE, so the comment after it is needed, as code has none.
        if (isKeyword(first, Set.of("drop", "truncate")) && tokens.size() == 4) {
            return tokens.get(1).kind() == 'v' && tokens.get(2).kind() == ';' && tokens.get(3).kind() == 'c'
                    && commentEndsValue(tokens.get(3).text());
        }
        return false;
    }

    /** True when the tokens contain UNION [ALL|DISTINCT] [(] SELECT followed by something. */
    private static boolean hasUnionSelect(List<Token> tokens) {
        for (int i = 0; i < tokens.size(); i++) {
            if (!isKeyword(tokens.get(i), Set.of("union"))) continue;
            int next = i + 1;
            while (next < tokens.size() && (tokens.get(next).kind() == '(' || tokens.get(next).text().equals("all")
                    || tokens.get(next).text().equals("distinct"))) {
                next++;
            }
            if (next + 1 < tokens.size() && isKeyword(tokens.get(next), Set.of("select")) && tokens.get(next + 1).kind() != 'c') {
                return true;
            }
        }
        return false;
    }

    /** A statement keyword after the value ends must start a plausible statement, as in ; DROP TABLE users. */
    private static boolean statement(List<Token> tokens) {
        if (tokens.isEmpty()) return false;
        Token first = tokens.get(0);
        if (isKeyword(first, Set.of("set"))) {
            // SET needs an assignment, as in ; SET @a = 1, not text such as BASE; SET SWAROVSKI.
            boolean assigns = tokens.stream().anyMatch(t -> t.kind() == 'o' && (t.text().equals("=") || t.text().equals(":=")));
            return assigns && valid(tokens);
        }
        if (isKeyword(first, Set.of("exec", "execute"))) {
            boolean nameOk = tokens.size() > 1 && (tokens.get(1).kind() == 'v' || tokens.get(1).kind() == 'f');
            boolean thenOk = tokens.size() < 3 || tokens.get(2).kind() != 'k';
            return nameOk && thenOk && valid(tokens);
        }
        return valid(tokens);
    }

    /** Checks what follows the end of the value: it must continue the query as SQL. */
    private static boolean breaksOut(List<Token> tokens, boolean numeric, boolean commentOk) {
        int index = 0;
        while (index < tokens.size() && tokens.get(index).kind() == ')') index++;
        if (index >= tokens.size()) return false;
        Token first = tokens.get(index);
        List<Token> rest = tokens.subList(index + 1, tokens.size());
        // As in admin'-- : the rest of the query is commented out. With an even number of
        // quotes, as in .hex('#C0FFEE'), the value closes its own string, so it's text.
        if (first.kind() == 'c') return !numeric && commentOk && commentEndsValue(first.text());
        if (first.kind() == ';') return !rest.isEmpty() && isKeyword(rest.get(0), STATEMENTS) && statement(rest);
        if (isKeyword(first, BARE_STATEMENTS)) return statement(tokens.subList(index, tokens.size()));
        // A column alias and FROM, as in ) AS orders FROM. After a number the alias must be
        // quoted, so "5 as gift from" stays text. After a bracket, as in ) AS t FROM, it needn't be.
        if (isKeyword(first, Set.of("as"))) {
            return rest.size() > 1 && isKeyword(rest.get(1), Set.of("from"))
                    && (rest.get(0).kind() == 's' || (rest.get(0).kind() == 'v' && (!numeric || index > 0)));
        }
        // NOT LIKE, NOT IN and so on, as in id'not like id--.
        if (isKeyword(first, Set.of("not")) && !rest.isEmpty() && isKeyword(rest.get(0), CONNECTORS)) {
            return breaksOut(rest, numeric, commentOk);
        }
        boolean connector = isKeyword(first, CONNECTORS)
                || (!numeric && first.kind() == 'o' && OPERATOR_CONNECTORS.contains(first.text()));
        if (!connector) {
            // A binary operator and a value count when a comment then hides the rest, as in
            // x'|email--. Not ~ or !, which take no left side, and not ASCII art such as '-----.
            return !numeric && commentOk && first.kind() == 'o'
                    && !first.text().equals("~") && !first.text().equals("!")
                    && !rest.isEmpty() && "nsvf(".indexOf(rest.get(0).kind()) >= 0
                    && endsWithHidingComment(rest) && valid(rest);
        }
        if (hasUnionSelect(rest)) return true;
        if (first.text().equals("union")) {
            Token next = rest.stream()
                    .filter(t -> t.kind() != '(' && !t.text().equals("all") && !t.text().equals("distinct"))
                    .findFirst().orElse(null);
            if (next == null || !isKeyword(next, Set.of("select"))) return false;
            // UNION SELECT is decisive, whatever its column list holds, as in from table/*.
            return rest.stream().filter(t -> t.kind() != 'c').count() >= 2;
        }
        if ((first.text().equals("order") || first.text().equals("group"))
                && (rest.isEmpty() || !isKeyword(rest.get(0), Set.of("by")))) {
            return false;
        }
        // ORDER BY and a column number counts columns for a UNION, as in -8970) ORDER BY 3631.
        if (first.text().equals("order") && rest.size() > 1 && rest.get(1).kind() == 'n' && valid(rest)) return true;
        // After a bare number, the connector alone isn't enough: "1 like this" is valid SQL too.
        return valid(rest) && strong(numeric ? rest : tokens.subList(index, tokens.size()));
    }

    private static int closingQuote(int[] chars, int quote, boolean backslash) {
        int index = 0;
        while (index < chars.length) {
            int c = chars[index];
            if (backslash && c == '\\') index += 2;
            else if (c == quote && index + 1 < chars.length && chars[index + 1] == quote) index += 2;
            else if (c == quote) return index;
            else index++;
        }
        return -1;
    }

    /**
     * True for a quoted SQL word in prose or code, as in {@code .method('is')}. The quote opens after a
     * space or bracket and closes straight after one keyword.
     */
    private static boolean quotedWord(int[] chars, int end, int quote) {
        // + is a space in form-encoded text, as in Use+'in'+or+'between', and Markdown puts code
        // in backticks, as in a `'limit'` event.
        if (end > 0 && " +([{,:=`".indexOf(chars[end - 1]) < 0) return false;
        int stop = end + 1;
        while (stop < chars.length && letter(chars[stop])) stop++;
        // The word closes and text follows, as in "IS" STAMP, not more SQL, as in 'or'1'='1.
        boolean followedByText = stop + 1 >= chars.length || Character.isWhitespace(chars[stop + 1])
                || "),.;:!?]}+`".indexOf(chars[stop + 1]) >= 0;
        return stop < chars.length && chars[stop] == quote && followedByText
                && KEYWORDS.contains(text(chars, end + 1, stop).toLowerCase(java.util.Locale.ROOT));
    }

    /** Returns a rule name when the value breaks out of a SQL value, otherwise null. */
    public static String classify(String value) {
        String text = clean(value);
        List<Token> tokens = lex(text);
        // A value that starts by closing a call or statement, as in ) UNION SELECT or ; DROP TABLE.
        if (!tokens.isEmpty() && (tokens.get(0).kind() == ')' || tokens.get(0).kind() == ';') && breaksOut(tokens, true, true)) {
            return "sql_breakout_numeric";
        }
        int start = tokens.size() > 1 && tokens.get(0).kind() == 'o'
                && (tokens.get(0).text().equals("-") || tokens.get(0).text().equals("+")) ? 1 : 0;
        if (tokens.size() > start && tokens.get(start).kind() == 'n') {
            // The number can be an expression, as in -6863 union or 189-999.9 union.
            while (start + 2 < tokens.size() && tokens.get(start + 1).kind() == 'o'
                    && "+-*/%".contains(tokens.get(start + 1).text()) && tokens.get(start + 2).kind() == 'n') {
                start += 2;
            }
            List<Token> after = tokens.subList(start + 1, tokens.size());
            if (breaksOut(after, true, true)) return "sql_breakout_numeric";
            if (after.size() > 2 && after.get(0).kind() == 'o' && "+-*/%".contains(after.get(0).text())
                    && after.get(1).kind() == '(' && isKeyword(after.get(2), Set.of("select")) && valid(after.subList(1, after.size()))) {
                return "sql_breakout_numeric";
            }
        }
        if (startsWithConnector(tokens) || statementValue(tokens) || STATEMENT_SHAPE.matcher(text).find()) {
            return "sql_breakout_numeric";
        }
        // A bare word can be the value too, as in foo) UNION ALL SELECT or foo WHERE 1=1 UNION ...
        if (!tokens.isEmpty() && tokens.get(0).kind() == 'v' && wordBreaksOut(tokens.subList(1, tokens.size()))) {
            return "sql_breakout_numeric";
        }
        // A word plus a subquery, as in foo + (SELECT 1 WHERE 2=2). The subquery needs FROM, WHERE
        // or a comparison, so "cheese + (select a topping)" stays text.
        if (tokens.size() > 3 && tokens.get(0).kind() == 'v' && tokens.get(1).kind() == 'o'
                && Set.of("+", "-", "||").contains(tokens.get(1).text()) && tokens.get(2).kind() == '('
                && isKeyword(tokens.get(3), Set.of("select")) && valid(tokens.subList(2, tokens.size()))
                && tokens.subList(3, tokens.size()).stream().anyMatch(t -> isKeyword(t, Set.of("from", "where"))
                        || (t.kind() == 'o' && COMPARISONS.contains(t.text())))) {
            return "sql_breakout_numeric";
        }
        int[] chars = text.codePoints().toArray();
        String[][] quotes = {{"'", "sql_breakout_single_quote"}, {"\"", "sql_breakout_double_quote"}};
        for (String[] quote : quotes) {
            if (!text.contains(quote[0])) continue;
            int q = quote[0].charAt(0);
            boolean oddCount = text.chars().filter(c -> c == q).count() % 2 == 1;
            String trimmed = text.stripLeading();
            if (oddCount && trimmed.startsWith(quote[0])) {
                // The value is a quoted string compared with the query's own, as in 'x'!='y.
                List<Token> whole = lex(text + quote[0]);
                if (!whole.isEmpty() && whole.get(0).kind() == 's' && breaksOut(whole.subList(1, whole.size()), false, true)) {
                    return quote[1];
                }
                // UNION or ALTER straight after the opening quote, as in 'UNION A or ' alter/.
                String after = trimmed.substring(1).stripLeading().toLowerCase(java.util.Locale.ROOT);
                for (String word : new String[]{"union", "alter"}) {
                    if (after.startsWith(word) && after.length() > word.length()
                            && (Character.isWhitespace(after.charAt(word.length())) || after.charAt(word.length()) == '/')) {
                        return quote[1];
                    }
                }
            }
            // MySQL treats a backslash as an escape and most other databases don't, so try both.
            boolean[] modes = text.indexOf('\\') >= 0 ? new boolean[]{true, false} : new boolean[]{false};
            for (boolean backslash : modes) {
                int end = closingQuote(chars, quote[0].charAt(0), backslash);
                if (end < 0 || quotedWord(chars, end, quote[0].charAt(0))) continue;
                boolean odd = text.chars().filter(c -> c == quote[0].charAt(0)).count() % 2 == 1;
                // The query's own closing quote follows the value.
                List<Token> tail = lex(text(chars, end + 1, chars.length) + quote[0]);
                if (breaksOut(tail, false, odd)) return quote[1];
                // Filters are dodged by dropping spaces, as in a'or'1'='1. Prose has a space after
                // the quote, as in "parents' or yours", so a glued connector is evidence enough.
                boolean glued = end + 1 < chars.length && Character.isLetter(chars[end + 1]);
                if (glued && tail.size() > 2 && isKeyword(tail.get(0), GLUED_CONNECTORS)
                        && valid(tail.subList(1, tail.size())) && tail.stream().noneMatch(t -> t.kind() == 'u')) {
                    return quote[1];
                }
                // The value ends the statement, as in admin'; or admin'/* x */;. Prose such as
                // "John's;" has a letter after the quote, and 18'; is a size in feet.
                if (odd && !(end > 0 && digit(chars[end - 1]))) {
                    List<Token> after = lex(text(chars, end + 1, chars.length));
                    if (after.stream().anyMatch(tk -> tk.kind() == ';') && after.stream().allMatch(tk -> tk.kind() == ';' || tk.kind() == 'c')) {
                        return quote[1];
                    }
                }
                // Blind injection often leaves the query's quote unmatched, as in ' and 526=527.
                if (odd && breaksOut(lex(text(chars, end + 1, chars.length)), true, false)) return quote[1];
            }
        }
        // Names that only appear in SQL, as in pg_sleep( or information_schema.
        if (SQL_ONLY_NAME.matcher(text).find() || POSTGRES_CAST.matcher(text).find() || sleepCall(text)) return "sql_function";
        if (value != null && COMMENT_OBFUSCATION.matcher(value).find()) return "sql_comment";
        return null;
    }

    private static boolean sleepCall(String text) {
        Matcher matcher = SLEEP_CALL.matcher(text);
        while (matcher.find()) {
            boolean bareLongDelay = matcher.group(3) != null && matcher.group(2).length() > 3
                    && text.trim().length() == matcher.group().length();
            if (matcher.group(1) == null && !bareLongDelay) return true;
        }
        return false;
    }

    /** Checks a value, and each name and value in it when it's a raw query string or form body. */
    public static String classifyFields(String value) {
        String rule = classify(value);
        if (rule != null || value == null || value.indexOf('=') < 0) return rule;
        String[] parts = value.split("&", 513);
        for (int i = 0; i < Math.min(parts.length, 512); i++) {
            int at = parts[i].indexOf('=');
            rule = classify(at < 0 ? parts[i] : parts[i].substring(0, at));
            if (rule == null && at >= 0) rule = classify(parts[i].substring(at + 1));
            if (rule != null) return rule;
        }
        return null;
    }
}
