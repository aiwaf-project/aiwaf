package com.aiwaf.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
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
    private static final Set<String> CONNECTORS = words("and between collate div group having in into is like limit or order procedure regexp rlike sounds union xor");
    // < and > are left out: quoted HTML such as ('<script>') would read as a comparison.
    private static final Set<String> OPERATOR_CONNECTORS = words("= <> != <=> || && +");
    private static final Set<String> COMPARISONS = words("= <> != < > <= >= <=>");
    private static final Set<String> STATEMENTS = words("alter create declare delete drop exec execute insert select set shutdown truncate update waitfor");
    private static final Set<String> STRONG_KEYWORDS = words("between exists group having in into is like limit order procedure regexp rlike select union");
    private static final Set<String> VALUE_KEYWORDS = words("false null true");
    private static final Set<String> END_KEYWORDS = words("asc desc end false null true");
    private static final List<String> MULTI_OPERATORS = List.of("<=>", "<>", "!=", "<=", ">=", "||", "&&", "::", ":=");
    private static final String SINGLE_OPERATORS = "=<>+-*/%&|^~!";
    private static final Set<Integer> WHITESPACE = Set.of(0x20, 0x09, 0x0a, 0x0d, 0x0b, 0x0c, 0xa0, 0x1680, 0x2000, 0x2001,
            0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000, 0xfeff);
    private static final Pattern VERSIONED_COMMENT = Pattern.compile("/\\*!\\d{0,6}([\\s\\S]*?)\\*/");
    private static final Pattern COMMENT = Pattern.compile("/\\*[\\s\\S]*?\\*/");

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
        text.codePoints().forEach(c -> out.appendCodePoint(WHITESPACE.contains(c) ? ' ' : c));
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
                tokens.add(new Token('c', ""));
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
                int after = end;
                while (after < length && chars[after] == ' ') after++;
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
        for (Token token : tokens) {
            if (token.kind() == 'c') break;
            if (token.kind() == 'x' || token.kind() == 'u') return false;
            if (last != null && last.kind() == 'f' && token.kind() != '(') return false;
            boolean atom = isAtom(token);
            if (atom && previousAtom) return false;
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

    /** Checks what follows the end of the value: it must continue the query as SQL. */
    private static boolean breaksOut(List<Token> tokens, boolean numeric, boolean commentOk) {
        int index = 0;
        while (index < tokens.size() && tokens.get(index).kind() == ')') index++;
        if (index >= tokens.size()) return false;
        Token first = tokens.get(index);
        List<Token> rest = tokens.subList(index + 1, tokens.size());
        // As in admin'-- : the rest of the query is commented out. With an even number of
        // quotes, as in .hex('#C0FFEE'), the value closes its own string, so it's text.
        if (first.kind() == 'c') return !numeric && commentOk;
        if (first.kind() == ';') return !rest.isEmpty() && isKeyword(rest.get(0), STATEMENTS);
        boolean connector = isKeyword(first, CONNECTORS)
                || (!numeric && first.kind() == 'o' && OPERATOR_CONNECTORS.contains(first.text()));
        if (!connector) return false;
        if (first.text().equals("union")) {
            Token next = rest.stream()
                    .filter(t -> t.kind() != '(' && !t.text().equals("all") && !t.text().equals("distinct"))
                    .findFirst().orElse(null);
            if (next == null || !isKeyword(next, Set.of("select"))) return false;
        }
        if ((first.text().equals("order") || first.text().equals("group"))
                && (rest.isEmpty() || !isKeyword(rest.get(0), Set.of("by")))) {
            return false;
        }
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
        // + is a space in form-encoded text, as in Use+'in'+or+'between'.
        if (end == 0 || " +([{,:=".indexOf(chars[end - 1]) < 0) return false;
        int stop = end + 1;
        while (stop < chars.length && letter(chars[stop])) stop++;
        return stop < chars.length && chars[stop] == quote
                && KEYWORDS.contains(text(chars, end + 1, stop).toLowerCase(java.util.Locale.ROOT));
    }

    /** Returns a rule name when the value breaks out of a SQL value, otherwise null. */
    public static String classify(String value) {
        String text = clean(value);
        List<Token> tokens = lex(text);
        int start = tokens.size() > 1 && tokens.get(0).kind() == 'o'
                && (tokens.get(0).text().equals("-") || tokens.get(0).text().equals("+")) ? 1 : 0;
        if (tokens.size() > start && tokens.get(start).kind() == 'n'
                && breaksOut(tokens.subList(start + 1, tokens.size()), true, true)) {
            return "sql_breakout_numeric";
        }
        int[] chars = text.codePoints().toArray();
        String[][] quotes = {{"'", "sql_breakout_single_quote"}, {"\"", "sql_breakout_double_quote"}};
        for (String[] quote : quotes) {
            if (!text.contains(quote[0])) continue;
            // MySQL treats a backslash as an escape and most other databases don't, so try both.
            boolean[] modes = text.indexOf('\\') >= 0 ? new boolean[]{true, false} : new boolean[]{false};
            for (boolean backslash : modes) {
                int end = closingQuote(chars, quote[0].charAt(0), backslash);
                // The query's own closing quote follows the value.
                long count = text.chars().filter(c -> c == quote[0].charAt(0)).count();
                if (end >= 0 && !quotedWord(chars, end, quote[0].charAt(0)) && breaksOut(lex(text(chars, end + 1, chars.length) + quote[0]), false, count % 2 == 1)) {
                    return quote[1];
                }
            }
        }
        return null;
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
