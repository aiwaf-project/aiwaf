package com.aiwaf.core;

import com.sun.source.tree.AnnotationTree;
import com.sun.source.tree.ArrayAccessTree;
import com.sun.source.tree.AssignmentTree;
import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.BlockTree;
import com.sun.source.tree.ClassTree;
import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.CompoundAssignmentTree;
import com.sun.source.tree.ConditionalExpressionTree;
import com.sun.source.tree.EnhancedForLoopTree;
import com.sun.source.tree.ExpressionTree;
import com.sun.source.tree.IdentifierTree;
import com.sun.source.tree.IfTree;
import com.sun.source.tree.BreakTree;
import com.sun.source.tree.StatementTree;
import com.sun.source.tree.SwitchTree;
import com.sun.source.tree.UnaryTree;
import com.sun.source.tree.CaseTree;
import com.sun.source.tree.CatchTree;
import com.sun.source.tree.DoWhileLoopTree;
import com.sun.source.tree.ForLoopTree;
import com.sun.source.tree.WhileLoopTree;
import com.sun.source.tree.LambdaExpressionTree;
import com.sun.source.tree.LineMap;
import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.MemberSelectTree;
import com.sun.source.tree.MethodInvocationTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.NewArrayTree;
import com.sun.source.tree.NewClassTree;
import com.sun.source.tree.ParenthesizedTree;
import com.sun.source.tree.ReturnTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.TypeCastTree;
import com.sun.source.tree.VariableTree;
import com.sun.source.util.JavacTask;
import com.sun.source.util.SourcePositions;
import com.sun.source.util.TreeScanner;
import com.sun.source.util.Trees;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/**
 * Reads Java source to find routes and request input that reaches dangerous operations.
 *
 * <p>It parses files only: nothing is compiled, loaded or run, and it doesn't affect request
 * handling. Calls into other classes in the scanned source are followed, through field, local and
 * parameter types, static calls and methods' return types, into every implementation of an interface. A clean report doesn't
 * prove code is safe, because reflection, dynamic dispatch and library code can be missed. This needs the JDK's compiler module, so a plain JRE can't run it.
 */
public final class JavaSourceAnalysisCore {
    public static final String SCHEMA_VERSION = "1.0";

    private static final Set<String> SKIPPED_DIRECTORIES = Set.of(".git", ".gradle", ".idea", "build", "node_modules", "out", "target");
    private static final long MAX_FILE_BYTES = 2L * 1024L * 1024L;
    private static final int MAX_HELPER_DEPTH = 3;
    // An interface with more implementations than this is followed into none of them.
    private static final int MAX_IMPLEMENTATIONS = 5;

    private static final Set<String> INPUT_ANNOTATIONS = Set.of(
            "Argument", "CookieValue", "CookieParam", "DestinationVariable", "FormParam", "HeaderParam", "InputArgument",
            "MatrixParam", "MatrixVariable", "ModelAttribute", "PathParam", "PathVariable", "Payload", "QueryParam", "RequestBody",
            "RequestHeader", "RequestParam", "RequestPart");
    // Methods called with a WebSocket or STOMP message, whose plain parameters are the message.
    private static final Set<String> MESSAGE_HANDLERS = Set.of("MessageMapping", "OnMessage", "SubscribeMapping");
    private static final Set<String> NOT_MESSAGE_TYPES = Set.of(
            "Message", "MessageHeaders", "Principal", "Session", "SimpMessageHeaderAccessor", "StompHeaderAccessor", "WebSocketSession");
    private static final Set<String> WHOLE_BODY_ANNOTATIONS = Set.of("ModelAttribute", "RequestBody", "RequestPart");
    private static final Set<String> REQUEST_TYPES = Set.of(
            "HttpServletRequest", "NativeWebRequest", "ServerHttpRequest", "ServerRequest", "ServletRequest", "WebRequest");
    private static final Set<String> REQUEST_METHODS = Set.of(
            "getCookies", "getHeader", "getHeaderNames", "getHeaders", "getInputStream", "getParameter", "getParameterMap", "getParameterNames",
            "getParameterValues", "getPart", "getParts", "getPathInfo", "getQueryString", "getReader", "getRequestURI", "getRequestURL",
            // Spring WebFlux ServerRequest and ServerHttpRequest.
            "bodyToFlux", "bodyToMono", "cookies", "formData", "getBody", "getQueryParams", "getURI", "headers", "multipartData",
            "pathVariable", "pathVariables", "queryParam", "queryParams");
    // Spring converts these before the method runs, so their values can't carry an injection.
    private static final Set<String> SAFE_TYPES = Set.of(
            "BigDecimal", "BigInteger", "Boolean", "Byte", "Double", "Duration", "Float", "Instant", "Integer", "LocalDate",
            "LocalDateTime", "LocalTime", "Long", "OffsetDateTime", "OptionalDouble", "OptionalInt", "OptionalLong", "Short", "UUID",
            "ZonedDateTime", "boolean", "byte", "double", "float", "int", "long", "short");
    private static final Set<String> SAFE_CONTAINERS = Set.of("Collection", "List", "Optional", "Set");
    private static final Set<String> SANITIZERS = Set.of(
            "Boolean.parseBoolean", "Boolean.valueOf", "Double.parseDouble", "Double.valueOf", "FilenameUtils.getName",
            "HtmlUtils.htmlEscape", "Integer.parseInt", "Integer.valueOf", "Long.parseLong", "Long.valueOf", "UUID.fromString");
    // Encoders whose result is safe only for some operations, as encodeForHTML is for HTML.
    private static final Map<String, Set<String>> KIND_SANITIZERS = Map.ofEntries(
            Map.entry("encodeForHTML", Set.of("xss")), Map.entry("encodeForHTMLAttribute", Set.of("xss")),
            Map.entry("encodeForJavaScript", Set.of("xss")), Map.entry("htmlEscape", Set.of("xss")),
            Map.entry("escapeHtml4", Set.of("xss")), Map.entry("escapeHtml", Set.of("xss")), Map.entry("forHtml", Set.of("xss")),
            Map.entry("forHtmlContent", Set.of("xss")), Map.entry("forHtmlAttribute", Set.of("xss")),
            Map.entry("encodeForLDAP", Set.of("ldap")), Map.entry("encodeForDN", Set.of("ldap")),
            Map.entry("encodeForXPath", Set.of("xpath")), Map.entry("encodeForSQL", Set.of("sql")),
            Map.entry("encodeForOS", Set.of("command")), Map.entry("quote", Set.of("redos")),
            Map.entry("encodeForURL", Set.of("redirect", "header")));
    // Marks the operations a source was made safe for: source + SAFE + "xss,ldap".
    private static final char SAFE = '\u0001';
    private static final Set<String> RESPONSE_WRITER_TYPES = Set.of("JspWriter", "PrintWriter", "ServletOutputStream");
    private static final Set<String> RESPONSE_WRITER_METHODS = Set.of("append", "format", "print", "printf", "println", "write");
    private static final Set<String> XML_PARSE_METHODS = Set.of("createXMLEventReader", "createXMLStreamReader", "parse", "read", "unmarshal");
    // Settings in a method that stop its XML parsers loading external entities.
    private static final List<String> XML_HARDENING = List.of(
            "disallow-doctype-decl", "external-general-entities", "FEATURE_SECURE_PROCESSING", "ACCESS_EXTERNAL_DTD",
            "IS_SUPPORTING_EXTERNAL_ENTITIES", "SUPPORT_DTD", "setExpandEntityReferences(false)");

    static String safeSource(String source, Set<String> kinds) {
        if (source == null || kinds.isEmpty()) {
            return source;
        }
        Set<String> all = new java.util.TreeSet<>(kinds);
        return baseSource(source) + SAFE + String.join(",", all);
    }

    static String baseSource(String source) {
        int marker = source.indexOf(SAFE);
        return marker < 0 ? source : source.substring(0, marker);
    }

    static Set<String> safeKinds(String source) {
        int marker = source.indexOf(SAFE);
        return marker < 0 ? Set.of() : Set.of(source.substring(marker + 1).split(","));
    }

    /** Two inputs in one value: it's safe only for what both are safe for. */
    static String combine(String left, String right) {
        if (left == null || right == null) {
            return left == null ? right : left;
        }
        Set<String> kinds = new HashSet<>(safeKinds(left));
        kinds.retainAll(safeKinds(right));
        return kinds.isEmpty() ? baseSource(safeKinds(left).isEmpty() ? left : right) : safeSource(left, kinds);
    }

    private static final Set<String> PASSTHROUGH_STATIC = Set.of(
            "Arrays.asList", "Base64.decodeBase64", "Base64.encodeBase64", "Base64.encodeBase64String", "List.of", "MessageFormat.format", "Objects.toString", "Optional.of", "Optional.ofNullable",
            "Set.of", "String.format", "String.join", "String.valueOf", "URLDecoder.decode");
    private static final Set<String> ARGUMENT_PASSTHROUGH = Set.of(
            "append", "concat", "decode", "encode", "encodeToString", "formatted", "insert", "join", "replace", "replaceAll", "replaceFirst", "resolve");
    private static final Set<String> NON_TEXT_METHODS = Set.of(
            "compareTo", "contains", "endsWith", "equals", "equalsIgnoreCase", "hashCode", "isBlank", "isEmpty", "length", "matches", "size", "startsWith");
    private static final Set<String> COLLECTING_METHODS = Set.of("add", "addAll", "append", "put", "putAll");
    private static final Set<String> SQL_ANY_RECEIVER = Set.of("createNativeQuery", "createQuery", "nativeQuery", "prepareCall", "prepareStatement");
    private static final Set<String> SQL_STATEMENT = Set.of("addBatch", "execute", "executeLargeUpdate", "executeQuery", "executeUpdate");
    private static final Set<String> SQL_TEMPLATE = Set.of("batchUpdate", "execute", "query", "queryForList", "queryForMap", "queryForObject", "queryForRowSet", "queryForStream", "update");
    private static final Set<String> PATH_CONSTRUCTORS = Set.of(
            "File", "FileInputStream", "FileOutputStream", "FileReader", "FileSystemResource", "FileWriter", "RandomAccessFile");
    private static final Set<String> SSRF_REST_METHODS = Set.of("exchange", "getForEntity", "getForObject", "postForEntity", "postForObject");

    private static final List<String> SPRING_MAPPINGS = List.of("DeleteMapping", "GetMapping", "PatchMapping", "PostMapping", "PutMapping", "RequestMapping");
    private static final Map<String, String> SERVLET_METHODS = Map.of(
            "doDelete", "DELETE", "doGet", "GET", "doHead", "HEAD", "doOptions", "OPTIONS", "doPatch", "PATCH", "doPost", "POST", "doPut", "PUT");

    private JavaSourceAnalysisCore() {}

    /** entryFile names the handler's file when the finding is in another file, and is empty otherwise. */
    public record Finding(String file, long line, String function, String sink, String call, String source, String confidence,
                          String entryFile) {
        public Finding(String file, long line, String function, String sink, String call, String source, String confidence) {
            this(file, line, function, sink, call, source, confidence, "");
        }
    }

    public record Route(String path, List<String> methods, String view, String name, ApiDetection api, Map<String, Object> auth) {}

    /** suppressed counts findings left out by aiwaf: ignore comments, and baselined those in a baseline. */
    public record Report(int filesScanned, List<Finding> findings, List<Map<String, String>> errors, List<Route> routes,
                         int suppressed, int baselined, List<JavaInputSchemaCore.Inputs> inputs) {
        public Report(int filesScanned, List<Finding> findings, List<Map<String, String>> errors, List<Route> routes) {
            this(filesScanned, findings, errors, routes, 0, 0, List.of());
        }

        public Report(int filesScanned, List<Finding> findings, List<Map<String, String>> errors, List<Route> routes,
                      int suppressed, int baselined) {
            this(filesScanned, findings, errors, routes, suppressed, baselined, List.of());
        }

        public Map<String, Object> toMap() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("schema_version", SCHEMA_VERSION);
            out.put("language", "java");
            out.put("files_scanned", filesScanned);
            List<Map<String, Object>> items = new ArrayList<>();
            for (Finding finding : findings) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("file", finding.file());
                item.put("line", finding.line());
                item.put("function", finding.function());
                item.put("sink", finding.sink());
                item.put("call", finding.call());
                item.put("source", finding.source());
                item.put("confidence", finding.confidence());
                if (finding.entryFile() != null && !finding.entryFile().isEmpty()) {
                    item.put("entry_file", finding.entryFile());
                }
                items.add(item);
            }
            out.put("findings", items);
            out.put("errors", errors);
            List<Map<String, Object>> handlers = new ArrayList<>();
            for (JavaInputSchemaCore.Inputs entry : inputs) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("function", entry.function());
                item.put("file", entry.file());
                item.put("inputs", entry.inputs());
                item.put("complete", entry.complete());
                handlers.add(item);
            }
            out.put("inputs", handlers);
            if (suppressed > 0) {
                out.put("suppressed", suppressed);
            }
            if (baselined > 0) {
                out.put("baselined", baselined);
            }
            return out;
        }

        public String toText() {
            StringBuilder out = new StringBuilder();
            for (Finding finding : findings) {
                out.append(finding.file()).append(':').append(finding.line()).append(": ").append(finding.sink())
                        .append(" (").append(finding.confidence()).append(" confidence) ").append(finding.source())
                        .append(" reaches ").append(finding.call()).append(" in ").append(finding.function()).append('\n');
            }
            for (Map<String, String> error : errors) {
                out.append(error.get("file")).append(": not analyzed: ").append(error.get("error")).append('\n');
            }
            out.append("Scanned ").append(filesScanned).append(" files and found ").append(findings.size())
                    .append(" possible injection paths.");
            if (suppressed > 0) {
                out.append("\nLeft out ").append(suppressed).append(" marked with an aiwaf: ignore comment.");
            }
            if (baselined > 0) {
                out.append("\nLeft out ").append(baselined).append(" already in the baseline.");
            }
            if (findings.isEmpty()) {
                out.append("\nThis doesn't prove the code is safe, because static analysis can miss paths.");
            }
            return out.toString();
        }
    }

    /** Scans a Java file or folder. Throws IllegalStateException when no JDK compiler is available. */
    public static Report analyze(Path root, boolean includeTests) throws IOException {
        if (root == null || !Files.exists(root)) {
            throw new IOException("Path not found: " + root);
        }
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            throw new IllegalStateException("Source analysis needs a JDK. This Java runtime has no compiler.");
        }
        List<Map<String, String>> errors = new ArrayList<>();
        List<Path> files = javaFiles(root, includeTests, errors);
        if (files.isEmpty()) {
            return new Report(0, List.of(), errors, List.of());
        }
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        List<Finding> findings = new ArrayList<>();
        List<Route> routes = new ArrayList<>();
        Set<String> failed = new LinkedHashSet<>();
        List<JavaInputSchemaCore.Inputs> inputs = new ArrayList<>();
        int parsed = 0;
        try (StandardJavaFileManager manager = compiler.getStandardFileManager(diagnostics, null, StandardCharsets.UTF_8)) {
            Iterable<? extends JavaFileObject> units = manager.getJavaFileObjectsFromPaths(files);
            JavacTask task = (JavacTask) compiler.getTask(null, manager, diagnostics, List.of("-proc:none"), null, units);
            List<CompilationUnitTree> trees = new ArrayList<>();
            task.parse().forEach(trees::add);
            for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics.getDiagnostics()) {
                if (diagnostic.getKind() == Diagnostic.Kind.ERROR && diagnostic.getSource() != null) {
                    String display = Path.of(diagnostic.getSource().toUri()).toString().replace('\\', '/');
                    if (failed.add(display)) {
                        errors.add(Map.of("file", display, "error", diagnostic.getMessage(Locale.ROOT)));
                    }
                }
            }
            SourcePositions positions = Trees.instance(task).getSourcePositions();
            // Every class is indexed first, so calls into other classes can be followed.
            Project project = new Project();
            List<UnitAnalyzer> analyzers = new ArrayList<>();
            for (CompilationUnitTree unit : trees) {
                String display = Path.of(unit.getSourceFile().toUri()).toString().replace('\\', '/');
                if (failed.contains(display)) {
                    continue;
                }
                parsed++;
                UnitAnalyzer analyzer = new UnitAnalyzer(unit, display, positions, findings, routes, project);
                analyzer.register();
                analyzers.add(analyzer);
            }
            // What each handler reads, with types, labelled as its findings are.
            List<CompilationUnitTree> readable = trees.stream()
                    .filter(unit -> !failed.contains(Path.of(unit.getSourceFile().toUri()).toString().replace('\\', '/'))).toList();
            inputs.addAll(JavaInputSchemaCore.inputs(readable, unit -> Path.of(unit.getSourceFile().toUri()).toString().replace('\\', '/')));
            // Again when input was found stored in a request or session attribute, for code read first.
            for (int pass = 0; pass < 3; pass++) {
                int before = project.attributes.size();
                analyzers.forEach(UnitAnalyzer::run);
                if (project.attributes.size() == before) {
                    break;
                }
                project.summaries.clear();
            }
        }
        findings.sort(Comparator.comparing(Finding::file).thenComparingLong(Finding::line).thenComparing(Finding::sink)
                .thenComparing(Finding::function));
        routes.sort(Comparator.comparing(Route::path));
        Map<String, List<String>> lines = new HashMap<>();
        List<Finding> kept = new ArrayList<>();
        for (Finding finding : findings) {
            List<String> text = lines.computeIfAbsent(finding.file(), file -> {
                try {
                    return Files.readAllLines(Path.of(file), StandardCharsets.UTF_8);
                } catch (IOException | RuntimeException ex) {
                    return List.of();
                }
            });
            if (!ignored(text, finding.line(), finding.sink())) {
                kept.add(finding);
            }
        }
        return new Report(parsed, List.copyOf(kept), List.copyOf(errors), List.copyOf(routes), findings.size() - kept.size(), 0,
                List.copyOf(inputs));
    }

    // aiwaf: ignore, or aiwaf: ignore[sql, xss], on a finding's line or the line before it.
    private static final java.util.regex.Pattern IGNORE =
            java.util.regex.Pattern.compile("aiwaf[:-]\\s*ignore(?:\\[([a-z_,\\s]+)\\])?", java.util.regex.Pattern.CASE_INSENSITIVE);

    static boolean ignored(List<String> lines, long line, String sink) {
        for (long number : new long[] {line, line - 1}) {
            if (number < 1 || number > lines.size()) {
                continue;
            }
            java.util.regex.Matcher match = IGNORE.matcher(lines.get((int) number - 1));
            if (match.find() && (match.group(1) == null
                    || java.util.Arrays.stream(match.group(1).split(",")).map(item -> item.trim().toLowerCase(Locale.ROOT)).anyMatch(sink::equals))) {
                return true;
            }
        }
        return false;
    }

    /** Identifies a finding across runs, without its line, which moves as code changes. */
    public static String fingerprint(String file, String sink, String call, String function, String source) {
        try {
            java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(String.join("\0", file, sink, call, function, source).getBytes(StandardCharsets.UTF_8));
            return java.util.HexFormat.of().formatHex(hash).substring(0, 32);
        } catch (java.security.NoSuchAlgorithmException ex) {
            throw new IllegalStateException(ex);
        }
    }

    public static String fingerprint(Finding finding) {
        return fingerprint(finding.file(), finding.sink(), finding.call(), finding.function(), finding.source());
    }

    /** Leaves out findings already in an earlier JSON report, so CI fails only on new ones. */
    public static Report applyBaseline(Report report, Map<String, Object> baseline) {
        Set<String> known = new HashSet<>();
        if (baseline.get("findings") instanceof List<?> items) {
            for (Object item : items) {
                if (item instanceof Map<?, ?> data) {
                    known.add(fingerprint(String.valueOf(data.get("file")), String.valueOf(data.get("sink")), String.valueOf(data.get("call")),
                            String.valueOf(data.get("function")), String.valueOf(data.get("source"))));
                }
            }
        }
        List<Finding> kept = report.findings().stream().filter(finding -> !known.contains(fingerprint(finding))).toList();
        return new Report(report.filesScanned(), kept, report.errors(), report.routes(), report.suppressed(),
                report.findings().size() - kept.size(), report.inputs());
    }

    private static final Map<String, String> SINK_TITLES = Map.ofEntries(
            Map.entry("code", "Code injection"), Map.entry("command", "Command injection"),
            Map.entry("deserialization", "Unsafe deserialization"), Map.entry("ldap", "LDAP injection"),
            Map.entry("nosql", "NoSQL injection"), Map.entry("path", "Path traversal"), Map.entry("prototype", "Prototype pollution"),
            Map.entry("redirect", "Open redirect"), Map.entry("redos", "Regular expression denial of service"),
            Map.entry("sql", "SQL injection"), Map.entry("ssrf", "Server-side request forgery"),
            Map.entry("template", "Template injection"), Map.entry("xpath", "XPath injection"),
            Map.entry("xss", "Cross-site scripting"), Map.entry("xxe", "XML external entity expansion"));

    /** The report as SARIF 2.1.0, for GitHub code scanning and other tools that read it. */
    public static Map<String, Object> toSarif(Report report) {
        List<String> sinks = report.findings().stream().map(Finding::sink).distinct().sorted().toList();
        List<Map<String, Object>> rules = new ArrayList<>();
        for (String sink : sinks) {
            String title = SINK_TITLES.getOrDefault(sink, sink);
            StringBuilder name = new StringBuilder();
            for (String word : title.split("[\\s-]+")) {
                name.append(Character.toUpperCase(word.charAt(0))).append(word.substring(1));
            }
            rules.add(Map.of("id", "aiwaf." + sink, "name", name.toString(), "shortDescription", Map.of("text", title),
                    "fullDescription", Map.of("text", "Request input can reach an operation where it allows " + title.toLowerCase(Locale.ROOT) + "."),
                    "defaultConfiguration", Map.of("level", "warning"), "properties", Map.of("tags", List.of("security"))));
        }
        List<Map<String, Object>> results = new ArrayList<>();
        for (Finding finding : report.findings()) {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("ruleId", "aiwaf." + finding.sink());
            result.put("ruleIndex", sinks.indexOf(finding.sink()));
            result.put("level", "high".equals(finding.confidence()) ? "error" : "warning");
            result.put("message", Map.of("text", finding.source() + " reaches " + finding.call() + " in " + finding.function()));
            String uri = finding.file().replace('\\', '/');
            result.put("locations", List.of(Map.of("physicalLocation", Map.of(
                    "artifactLocation", Map.of("uri", uri.startsWith("./") ? uri.substring(2) : uri),
                    "region", Map.of("startLine", Math.max(finding.line(), 1))))));
            result.put("partialFingerprints", Map.of("aiwafFinding/v1", fingerprint(finding)));
            results.add(result);
        }
        Map<String, Object> sarif = new LinkedHashMap<>();
        sarif.put("$schema", "https://json.schemastore.org/sarif-2.1.0.json");
        sarif.put("version", "2.1.0");
        sarif.put("runs", List.of(Map.of(
                "tool", Map.of("driver", Map.of("name", "AIWAF audit", "informationUri", "https://github.com/aiwaf-project/aiwaf", "rules", rules)),
                "results", results)));
        return sarif;
    }

    private static List<Path> javaFiles(Path root, boolean includeTests, List<Map<String, String>> errors) throws IOException {
        List<Path> files = new ArrayList<>();
        if (Files.isRegularFile(root)) {
            return List.of(root);
        }
        try (Stream<Path> paths = Files.walk(root)) {
            paths.filter(Files::isRegularFile).filter(path -> path.toString().endsWith(".java")).sorted().forEach(path -> {
                if (skipped(root.relativize(path), includeTests)) {
                    return;
                }
                try {
                    if (Files.size(path) > MAX_FILE_BYTES) {
                        errors.add(Map.of("file", path.toString().replace('\\', '/'), "error", "File is too large to analyze"));
                        return;
                    }
                } catch (IOException ex) {
                    errors.add(Map.of("file", path.toString().replace('\\', '/'), "error", String.valueOf(ex.getMessage())));
                    return;
                }
                files.add(path);
            });
        }
        return files;
    }

    private static boolean skipped(Path relative, boolean includeTests) {
        for (Path part : relative) {
            String name = part.toString();
            if (SKIPPED_DIRECTORIES.contains(name) || (!includeTests && name.equals("test"))) {
                return true;
            }
        }
        String fileName = relative.getFileName().toString();
        return !includeTests && (fileName.endsWith("Test.java") || fileName.endsWith("Tests.java") || fileName.endsWith("IT.java"));
    }

    /** True for parameter types Spring converts to numbers, booleans, dates or UUIDs, or collections of them. */
    static boolean safeType(Tree tree) {
        String text = String.valueOf(tree).replaceAll("@\\S+\\s+", "").replace(" ", "");
        while (text.endsWith("[]")) {
            text = text.substring(0, text.length() - 2);
        }
        int generic = text.indexOf('<');
        if (generic > 0 && text.endsWith(">")) {
            String outer = text.substring(text.lastIndexOf('.', generic) + 1, generic);
            String inner = text.substring(generic + 1, text.length() - 1);
            return SAFE_CONTAINERS.contains(outer) && SAFE_TYPES.contains(inner.substring(inner.lastIndexOf('.') + 1));
        }
        return SAFE_TYPES.contains(text.substring(text.lastIndexOf('.') + 1));
    }

    /** The scanned classes, by simple name, so calls can be followed across files. */
    private static final class Project {
        private final Map<String, ClassInfo> classes = new HashMap<>();
        private final Set<String> ambiguous = new HashSet<>();
        private final Map<String, List<ClassInfo>> subtypes = new HashMap<>();
        private final Map<ClassTree, ClassInfo> byTree = new IdentityHashMap<>();
        // Request and session attributes set from input, by name, for the code that reads them back.
        private final Map<String, String> attributes = new HashMap<>();
        // Helpers already walked with the same input: what they returned and found.
        private final Map<String, Object[]> summaries = new HashMap<>();
        private final List<List<Object[]>> recordings = new ArrayList<>();

        void add(ClassInfo info) {
            byTree.put(info.tree, info);
            if (ambiguous.contains(info.name)) {
                return;
            }
            if (classes.putIfAbsent(info.name, info) != null) {
                classes.remove(info.name);
                ambiguous.add(info.name);
            }
            List<Tree> parents = new ArrayList<>(info.tree.getImplementsClause());
            if (info.tree.getExtendsClause() != null) {
                parents.add(info.tree.getExtendsClause());
            }
            for (Tree parent : parents) {
                subtypes.computeIfAbsent(simpleName(parent), key -> new ArrayList<>()).add(info);
            }
        }

        /**
         * Finds the methods a call on this type may run: its own, or each implementation's, since any
         * of them may be the one injected.
         */
        List<Target> methods(UnitAnalyzer from, String type, String name, int arguments) {
            ClassInfo info = type == null ? null : from.local.getOrDefault(type, classes.get(type));
            if (info == null) {
                return List.of();
            }
            MethodTree own = info.method(name, arguments);
            if (own != null && own.getBody() != null) {
                return List.of(new Target(info, own));
            }
            List<Target> found = new ArrayList<>();
            for (ClassInfo subtype : subtypes.getOrDefault(type, List.of())) {
                MethodTree method = subtype.method(name, arguments);
                if (method != null && method.getBody() != null) {
                    found.add(new Target(subtype, method));
                }
            }
            return found.size() > MAX_IMPLEMENTATIONS ? List.of() : found;
        }
    }

    private record Target(ClassInfo owner, MethodTree method) {}

    private static final class ClassInfo {
        private final UnitAnalyzer unit;
        private final ClassTree tree;
        private final String name;
        private final Map<String, List<MethodTree>> methods = new HashMap<>();
        private final Map<String, String> fields = new HashMap<>();

        ClassInfo(UnitAnalyzer unit, ClassTree tree) {
            this.unit = unit;
            this.tree = tree;
            this.name = tree.getSimpleName().toString();
            for (Tree member : tree.getMembers()) {
                if (member instanceof MethodTree method) {
                    methods.computeIfAbsent(method.getName().toString(), key -> new ArrayList<>()).add(method);
                } else if (member instanceof VariableTree field) {
                    fields.put(field.getName().toString(), simpleName(field.getType()));
                }
            }
        }

        MethodTree method(String name, int arguments) {
            for (MethodTree method : methods.getOrDefault(name, List.of())) {
                if (method.getParameters().size() == arguments) {
                    return method;
                }
            }
            return null;
        }
    }

    private static String simpleName(Tree tree) {
        String text = String.valueOf(tree);
        int generic = text.indexOf('<');
        if (generic >= 0) {
            text = text.substring(0, generic);
        }
        return text.substring(text.lastIndexOf('.') + 1).trim();
    }

    private static final class UnitAnalyzer {
        private final CompilationUnitTree unit;
        private final String file;
        private final SourcePositions positions;
        private final List<Finding> findings;
        private final List<Route> routes;
        private final Project project;
        // This file's own classes, which a name in this file means before any elsewhere.
        private final Map<String, ClassInfo> local = new HashMap<>();
        private final Set<String> seen = new HashSet<>();

        UnitAnalyzer(CompilationUnitTree unit, String file, SourcePositions positions, List<Finding> findings, List<Route> routes,
                     Project project) {
            this.unit = unit;
            this.file = file;
            this.positions = positions;
            this.findings = findings;
            this.routes = routes;
            this.project = project;
        }

        void register() {
            new TreeScanner<Void, Void>() {
                @Override
                public Void visitClass(ClassTree node, Void unused) {
                    ClassInfo info = new ClassInfo(UnitAnalyzer.this, node);
                    project.add(info);
                    local.putIfAbsent(info.name, info);
                    return super.visitClass(node, unused);
                }
            }.scan(unit, null);
        }

        // Routes come from the first pass only; later passes look again for input only.
        private boolean routesCollected;

        void run() {
            new TreeScanner<Void, Void>() {
                @Override
                public Void visitClass(ClassTree node, Void unused) {
                    analyzeClass(node);
                    return super.visitClass(node, unused);
                }
            }.scan(unit, null);
            routesCollected = true;
        }

        private void analyzeClass(ClassTree type) {
            ClassInfo owner = project.byTree.get(type);
            String className = type.getSimpleName().toString();
            for (Tree member : type.getMembers()) {
                if (!(member instanceof MethodTree method) || method.getBody() == null) {
                    continue;
                }
                Map<String, String> tainted = new HashMap<>();
                Set<String> requests = new HashSet<>();
                boolean messages = method.getModifiers().getAnnotations().stream()
                        .anyMatch(item -> MESSAGE_HANDLERS.contains(simpleName(item.getAnnotationType())));
                for (VariableTree parameter : method.getParameters()) {
                    AnnotationTree annotation = parameter.getModifiers().getAnnotations().stream()
                            .filter(item -> INPUT_ANNOTATIONS.contains(simpleName(item.getAnnotationType())))
                            .findFirst().orElse(null);
                    String name = parameter.getName().toString();
                    if (annotation == null && messages && parameter.getModifiers().getAnnotations().isEmpty()
                            && !NOT_MESSAGE_TYPES.contains(simpleName(parameter.getType())) && !safeType(parameter.getType())) {
                        tainted.put(name, "websocket message " + name);
                    } else if (annotation != null) {
                        if (!safeType(parameter.getType())) {
                            tainted.put(name, inputSource(annotation, name));
                        }
                    } else if (REQUEST_TYPES.contains(simpleName(parameter.getType()))) {
                        requests.add(name);
                    }
                }
                if (!tainted.isEmpty() || !requests.isEmpty()) {
                    analyzeMethod(owner, method, tainted, requests, className + "." + method.getName(), 0, new HashSet<>(), "");
                }
            }
            if (!routesCollected) {
                collectRoutes(type, className);
            }
        }

        /** A parameter's HTTP name can differ from its Java name. A whole-body parameter has none. */
        private static String inputSource(AnnotationTree annotation, String parameter) {
            String type = simpleName(annotation.getAnnotationType());
            if (WHOLE_BODY_ANNOTATIONS.contains(type)) {
                return "@" + type;
            }
            List<String> names = stringsFrom(annotation);
            return "@" + type + " " + (names.isEmpty() ? parameter : names.get(0));
        }

        /** Walks a method and returns the input source its return value carries, if any. */
        /**
         * Walks a method and returns the input source its return value carries, if any. A helper walked
         * before with the same input isn't walked again: its findings are recorded again for this caller.
         */
        @SuppressWarnings("unchecked")
        private static String analyzeMethod(ClassInfo owner, MethodTree method, Map<String, String> tainted, Set<String> requests,
                                            String label, int depth, Set<String> active, String entry) {
            String key = owner.name + "." + method.getName() + ":" + method.getParameters().size();
            if (depth > MAX_HELPER_DEPTH || method.getBody() == null || active.contains(key)) {
                return null;
            }
            Project project = owner.unit.project;
            String summaryKey = depth == 0 ? null : System.identityHashCode(method) + ":" + key + ":" + depth + ":"
                    + new java.util.TreeMap<>(tainted) + ":" + new java.util.TreeSet<>(requests);
            if (summaryKey != null && project.summaries.containsKey(summaryKey)) {
                Object[] summary = project.summaries.get(summaryKey);
                String before = (String) summary[0];
                for (Object[] item : (List<Object[]>) summary[1]) {
                    ((UnitAnalyzer) item[0]).report((Tree) item[1], label + ((String) item[2]).substring(before.length()),
                            (String) item[3], (String) item[4], (String) item[5], (String) item[6], entry);
                }
                return (String) summary[2];
            }
            List<Object[]> recorded = new ArrayList<>();
            if (summaryKey != null) {
                project.recordings.add(recorded);
            }
            active.add(key);
            MethodAnalyzer analyzer = new MethodAnalyzer(owner, method, tainted, requests, label, depth, active, entry);
            try {
                analyzer.scan(method.getBody().getStatements(), null);
            } finally {
                active.remove(key);
                if (summaryKey != null) {
                    project.recordings.remove(project.recordings.size() - 1);
                }
            }
            if (summaryKey != null) {
                project.summaries.put(summaryKey, new Object[] {label, recorded, analyzer.returned});
            }
            return analyzer.returned;
        }

        // entry is the file of the handler that called into this one, when it is another file.
        void report(Tree node, String function, String sink, String call, String source, String confidence, String entry) {
            for (List<Object[]> recorded : project.recordings) {
                recorded.add(new Object[] {this, node, function, sink, call, source, confidence});
            }
            long position = positions.getStartPosition(unit, node);
            LineMap lines = unit.getLineMap();
            long line = position >= 0 ? lines.getLineNumber(position) : 0;
            if (safeKinds(source).contains(sink) || safeKinds(source).contains("*")) {
                return;  // Encoded for this operation, as encodeForHTML is before HTML output.
            }
            if (seen.add(line + ":" + sink + ":" + call + ":" + function.split(" -> ")[0] + ":" + entry)) {
                findings.add(new Finding(file, line, function, sink, call, baseSource(source), confidence, entry));
            }
        }

        private void collectRoutes(ClassTree type, String className) {
            Map<String, AnnotationTree> annotations = annotations(type.getModifiers().getAnnotations());
            boolean rest = annotations.containsKey("RestController");
            if (rest || annotations.containsKey("Controller")) {
                List<String> prefixes = annotations.containsKey("RequestMapping")
                        ? stringsFrom(annotations.get("RequestMapping")) : List.of("");
                for (Tree member : type.getMembers()) {
                    if (member instanceof MethodTree method) {
                        springRoute(method, className, rest, prefixes.isEmpty() ? List.of("") : prefixes);
                    }
                }
            }
            if (annotations.containsKey("WebServlet")) {
                servletRoute(type, className, annotations.get("WebServlet"));
            }
        }

        private void springRoute(MethodTree method, String className, boolean rest, List<String> prefixes) {
            Map<String, AnnotationTree> annotations = annotations(method.getModifiers().getAnnotations());
            String mapping = SPRING_MAPPINGS.stream().filter(annotations::containsKey).findFirst().orElse(null);
            if (mapping == null) {
                return;
            }
            AnnotationTree annotation = annotations.get(mapping);
            List<String> paths = stringsFrom(annotation);
            List<String> methods = new ArrayList<>();
            if (mapping.equals("RequestMapping")) {
                for (ExpressionTree argument : annotation.getArguments()) {
                    if (argument instanceof AssignmentTree assignment && "method".equals(String.valueOf(assignment.getVariable()))) {
                        String text = String.valueOf(assignment.getExpression());
                        for (String name : List.of("GET", "POST", "PUT", "PATCH", "DELETE")) {
                            if (text.contains(name)) {
                                methods.add(name);
                            }
                        }
                    }
                }
                if (methods.isEmpty()) {
                    methods.addAll(List.of("GET", "POST"));
                }
            } else {
                methods.add(mapping.replace("Mapping", "").toUpperCase(Locale.ROOT));
            }
            Signals signals = new Signals();
            if (rest) {
                signals.api(50, "class:@RestController");
                signals.payload = "json";
            }
            if (annotations.containsKey("ResponseBody")) {
                signals.api(40, "method:@ResponseBody");
                signals.payload = "json";
            }
            for (VariableTree parameter : method.getParameters()) {
                Set<String> names = annotations(parameter.getModifiers().getAnnotations()).keySet();
                if (names.contains("RequestBody")) {
                    signals.api(30, "parameter:@RequestBody");
                    signals.payload = "json";
                    signals.requestBody = true;
                }
                if (names.contains("ModelAttribute") || (!rest && names.contains("RequestParam") && methods.contains("POST"))) {
                    signals.form(40, "parameter:form");
                    signals.requestBody = true;
                }
            }
            signals.readBody(method);
            for (String prefix : prefixes) {
                for (String path : paths.isEmpty() ? List.of("") : paths) {
                    addRoute(join(prefix, path), methods, className, method.getName().toString(), signals);
                }
            }
        }

        private void servletRoute(ClassTree type, String className, AnnotationTree annotation) {
            List<String> paths = stringsFrom(annotation);
            List<String> methods = new ArrayList<>();
            Signals signals = new Signals();
            for (Tree member : type.getMembers()) {
                if (member instanceof MethodTree method && method.getBody() != null) {
                    String name = method.getName().toString();
                    String httpMethod = SERVLET_METHODS.get(name);
                    if (httpMethod != null) {
                        methods.add(httpMethod);
                    }
                    if (httpMethod != null || name.equals("service")) {
                        signals.readBody(method);
                    }
                }
            }
            for (String path : paths) {
                addRoute(path, methods.isEmpty() ? List.of("GET", "POST") : methods, className, "", signals);
            }
        }

        private void addRoute(String rawPath, List<String> methods, String className, String methodName, Signals signals) {
            String path = PathManifestCore.normalizePath(rawPath);
            Map<String, Object> auth = new HashMap<>(AuthDetectionCore.detectAuthEndpoint(path, null, null));
            if (signals.login && !Boolean.TRUE.equals(auth.get("is_auth"))) {
                auth.put("is_auth", true);
                auth.put("auth_type", "login");
            }
            String view = methodName.isEmpty() ? className : className + "." + methodName;
            routes.add(new Route(path, List.copyOf(methods), view, methodName.isEmpty() ? className : methodName, signals.detection(path), auth));
        }

        private static Map<String, AnnotationTree> annotations(List<? extends AnnotationTree> items) {
            Map<String, AnnotationTree> out = new HashMap<>();
            for (AnnotationTree item : items) {
                out.put(simpleName(item.getAnnotationType()), item);
            }
            return out;
        }

        private static List<String> stringsFrom(AnnotationTree annotation) {
            List<String> values = new ArrayList<>();
            for (ExpressionTree argument : annotation.getArguments()) {
                if (argument instanceof AssignmentTree assignment) {
                    String name = String.valueOf(assignment.getVariable());
                    if (Set.of("value", "name", "path", "urlPatterns").contains(name)) {
                        addStrings(assignment.getExpression(), values);
                    }
                } else {
                    addStrings(argument, values);
                }
            }
            return values;
        }

        private static void addStrings(ExpressionTree expression, List<String> values) {
            if (expression instanceof LiteralTree literal && literal.getValue() instanceof String text) {
                values.add(text);
            } else if (expression instanceof NewArrayTree array && array.getInitializers() != null) {
                array.getInitializers().forEach(item -> addStrings(item, values));
            }
        }

        private static String join(String left, String right) {
            String first = left == null ? "" : left;
            String second = right == null ? "" : right;
            if (first.endsWith("/") && second.startsWith("/")) {
                return first + second.substring(1);
            }
            if (!first.isEmpty() && !first.endsWith("/") && !second.isEmpty() && !second.startsWith("/")) {
                return first + "/" + second;
            }
            return first + second;
        }
    }

    /** Route signals that only source can show, because reflection can't see method bodies. */
    private static final class Signals {
        private int apiScore;
        private int formScore;
        private final List<String> apiSignals = new ArrayList<>();
        private final List<String> formSignals = new ArrayList<>();
        private String payload = "";
        private boolean requestBody;
        private boolean login;

        void api(int score, String signal) {
            apiScore += score;
            apiSignals.add(signal);
        }

        void form(int score, String signal) {
            formScore += score;
            formSignals.add(signal);
        }

        void readBody(MethodTree method) {
            if (method.getBody() == null) {
                return;
            }
            new TreeScanner<Void, Void>() {
                @Override
                public Void visitMethodInvocation(MethodInvocationTree node, Void unused) {
                    if (node.getMethodSelect() instanceof MemberSelectTree select) {
                        String name = select.getIdentifier().toString();
                        String receiver = String.valueOf(select.getExpression()).toLowerCase(Locale.ROOT);
                        if (name.equals("readValue") || name.equals("readTree")) {
                            api(40, "body:json_parse");
                            payload = "json";
                            requestBody = true;
                        } else if (name.equals("getInputStream") || name.equals("getReader")) {
                            api(20, "body:request_stream");
                            requestBody = true;
                        } else if (name.equals("setContentType") && String.valueOf(node.getArguments()).contains("json")) {
                            api(40, "response:json");
                            payload = payload.isEmpty() ? "json" : payload;
                        } else if (name.equals("getParameter") || name.equals("getParameterMap")) {
                            form(20, "body:request_parameter");
                        } else if (name.equals("authenticate") && receiver.contains("auth")) {
                            login = true;
                        } else if (name.equals("login") && receiver.contains("request")) {
                            login = true;
                        } else if (name.equals("matches") && receiver.contains("encoder")) {
                            login = true;
                        }
                    }
                    return super.visitMethodInvocation(node, unused);
                }
            }.scan(method.getBody(), null);
        }

        ApiDetection detection(String path) {
            int score = apiScore;
            List<String> signals = new ArrayList<>(apiSignals);
            String lower = path.toLowerCase(Locale.ROOT);
            if (lower.startsWith("/api/")) {
                score += 30;
                signals.add("path:/api");
            }
            double confidence = Math.min(1.0, score / 100.0);
            double formConfidence = apiScore >= formScore && apiScore > 0 ? 0.0 : Math.min(1.0, formScore / 100.0);
            return new ApiDetection(confidence >= 0.3, payload.isEmpty() ? "" : "json", payload, confidence, List.copyOf(signals),
                    requestBody, formConfidence, formConfidence > 0.0 ? List.copyOf(formSignals) : List.of());
        }
    }

    private static final class MethodAnalyzer extends TreeScanner<Void, Void> {
        private final ClassInfo owner;
        private final UnitAnalyzer unit;
        private final Map<String, String> tainted;
        private final Set<String> requests;
        private final String label;
        private final int depth;
        private final Set<String> active;
        private final String entry;
        private final Map<String, String> types = new HashMap<>();
        private final Map<MethodInvocationTree, String> helperReturns = new IdentityHashMap<>();
        private String returned;
        private int nesting;
        private int lambdas;
        // The method turns off external entities for its XML parsers.
        private final boolean hardened;
        // Local values known before running, so if ((7 * 42) - num > 200) can be decided.
        private final Map<String, Object> constants = new HashMap<>();
        // Maps filled with literal keys, by key, and lists, by position: the input each holds, if any.
        private final Map<String, Map<Object, String>> keyed = new HashMap<>();
        private final Map<String, List<String>> lists = new HashMap<>();
        // What objects built here hold in their fields: input, or the request itself.
        private final Map<String, Map<String, String>> fieldInput = new HashMap<>();
        private final Map<String, Set<String>> fieldRequests = new HashMap<>();
        private static final Set<String> LIST_TYPES = Set.of("ArrayList", "LinkedList", "Vector", "CopyOnWriteArrayList");

        MethodAnalyzer(ClassInfo owner, MethodTree method, Map<String, String> tainted, Set<String> requests, String label,
                       int depth, Set<String> active, String entry) {
            String body = String.valueOf(method.getBody());
            this.hardened = XML_HARDENING.stream().anyMatch(body::contains);
            this.owner = owner;
            this.unit = owner.unit;
            this.tainted = new HashMap<>(tainted);
            this.requests = requests;
            this.label = label;
            this.depth = depth;
            this.active = active;
            this.entry = entry;
            types.putAll(owner.fields);
            for (VariableTree parameter : method.getParameters()) {
                types.put(parameter.getName().toString(), simpleName(parameter.getType()));
            }
        }

        @Override
        public Void visitReturn(ReturnTree node, Void unused) {
            super.visitReturn(node, unused);
            if (lambdas == 0 && returned == null && node.getExpression() != null) {
                returned = taint(node.getExpression());
            }
            return null;
        }

        @Override
        public Void visitLambdaExpression(LambdaExpressionTree node, Void unused) {
            lambdas++;
            super.visitLambdaExpression(node, unused);
            lambdas--;
            return null;
        }

        @Override
        public Void visitBlock(BlockTree node, Void unused) {
            nesting++;
            super.visitBlock(node, unused);
            nesting--;
            return null;
        }

        @Override
        public Void visitClass(ClassTree node, Void unused) {
            return null;
        }

        @Override
        public Void visitVariable(VariableTree node, Void unused) {
            super.visitVariable(node, unused);
            // var (which has no type before compiling) takes its class from the value: new UserService()
            // or a method's return type.
            String type = node.getType() == null ? "var" : simpleName(node.getType());
            if (type.equals("var")) {
                type = node.getInitializer() == null ? null : typeOf(node.getInitializer());
            }
            if (type != null) {
                types.put(node.getName().toString(), type);
            }
            String variable = node.getName().toString();
            Object known = nesting == 0 && node.getInitializer() != null ? constant(node.getInitializer()) : null;
            if (known != null) {
                constants.put(variable, known);
            } else {
                constants.remove(variable);
            }
            keyed.remove(variable);
            lists.remove(variable);
            remember(variable, node.getInitializer());
            if (node.getInitializer() instanceof NewClassTree construction && construction.getArguments().isEmpty()
                    && LIST_TYPES.contains(simpleName(construction.getIdentifier()))) {
                lists.put(variable, new ArrayList<>());
            }
            if (node.getInitializer() != null) {
                bind(node.getName().toString(), taint(node.getInitializer()));
            }
            return null;
        }

        @Override
        public Void visitAssignment(AssignmentTree node, Void unused) {
            super.visitAssignment(node, unused);
            String name = targetName(node.getVariable());
            forget(node.getVariable());
            if (name != null && node.getVariable() instanceof IdentifierTree) {
                keyed.remove(name);
                lists.remove(name);
                remember(name, node.getExpression());
            }
            if (name != null) {
                bind(name, taint(node.getExpression()));
            }
            return null;
        }

        @Override
        public Void visitCompoundAssignment(CompoundAssignmentTree node, Void unused) {
            super.visitCompoundAssignment(node, unused);
            forget(node.getVariable());
            String name = targetName(node.getVariable());
            String source = taint(node.getExpression());
            if (name != null && source != null) {
                tainted.put(name, source);
            }
            return null;
        }

        @Override
        public Void visitEnhancedForLoop(EnhancedForLoopTree node, Void unused) {
            String source = taint(node.getExpression());
            if (source != null) {
                tainted.put(node.getVariable().getName().toString(), source);
            }
            forgetAssignedIn(node.getStatement());
            return branch(() -> super.visitEnhancedForLoop(node, unused));
        }

        // Code that may not run, even without braces, as in if (param == null) param = "";
        private Void branch(java.util.function.Supplier<Void> visit) {
            nesting++;
            try {
                return visit.get();
            } finally {
                nesting--;
            }
        }

        @Override
        public Void visitIf(IfTree node, Void unused) {
            scan(node.getCondition(), unused);
            // A condition known before running decides which branch runs, and that branch always does.
            if (constant(node.getCondition()) instanceof Boolean known) {
                return scan(known ? node.getThenStatement() : node.getElseStatement(), unused);
            }
            Map<String, Set<String>> failed = validated(node.getCondition(), false);
            if (!failed.isEmpty() && node.getElseStatement() == null && exits(node.getThenStatement())) {
                // A check that leaves when it fails: what follows only sees values that passed.
                branch(() -> scan(node.getThenStatement(), unused));
                failed.forEach((name, kinds) -> tainted.computeIfPresent(name, (key, source) -> safeSource(source, kinds)));
                return null;
            }
            Map<String, Set<String>> passed = validated(node.getCondition(), true);
            if (!passed.isEmpty() && !exits(node.getThenStatement())) {
                Map<String, String> before = new HashMap<>();
                passed.forEach((name, kinds) -> {
                    if (tainted.containsKey(name)) {
                        before.put(name, tainted.get(name));
                        tainted.put(name, safeSource(tainted.get(name), kinds));
                    }
                });
                branch(() -> scan(node.getThenStatement(), unused));
                before.forEach((name, source) -> {
                    if (safeSource(source, passed.get(name)).equals(tainted.get(name))) {
                        tainted.put(name, source);
                    }
                });
                return branch(() -> scan(node.getElseStatement(), unused));
            }
            return branch(() -> {
                scan(node.getThenStatement(), unused);
                return scan(node.getElseStatement(), unused);
            });
        }

        private static boolean exits(StatementTree statement) {
            if (statement instanceof BlockTree block) {
                return !block.getStatements().isEmpty() && exits(block.getStatements().get(block.getStatements().size() - 1));
            }
            return statement instanceof ReturnTree || statement instanceof com.sun.source.tree.ThrowTree
                    || statement instanceof BreakTree || statement instanceof com.sun.source.tree.ContinueTree;
        }

        /**
         * The tainted names a condition shows are safe when it is outcome, and what for: regex
         * matches, allowlists, equality with a literal, numeric checks and '..' checks.
         */
        private Map<String, Set<String>> validated(ExpressionTree test, boolean outcome) {
            Map<String, Set<String>> found = new HashMap<>();
            if (test instanceof ParenthesizedTree parenthesized) {
                return validated(parenthesized.getExpression(), outcome);
            }
            if (test instanceof UnaryTree unary && unary.getKind() == Tree.Kind.LOGICAL_COMPLEMENT) {
                return validated(unary.getExpression(), !outcome);
            }
            if (test instanceof BinaryTree binary) {
                Tree.Kind kind = binary.getKind();
                if ((kind == Tree.Kind.CONDITIONAL_AND && outcome) || (kind == Tree.Kind.CONDITIONAL_OR && !outcome)) {
                    validated(binary.getLeftOperand(), outcome).forEach((name, kinds) -> found.computeIfAbsent(name, key -> new HashSet<>()).addAll(kinds));
                    validated(binary.getRightOperand(), outcome).forEach((name, kinds) -> found.computeIfAbsent(name, key -> new HashSet<>()).addAll(kinds));
                }
                return found;
            }
            if (!(test instanceof MethodInvocationTree call) || !(call.getMethodSelect() instanceof MemberSelectTree select)) {
                return found;
            }
            String method = select.getIdentifier().toString();
            ExpressionTree receiver = select.getExpression();
            List<? extends ExpressionTree> arguments = call.getArguments();
            String receiverName = targetName(receiver);
            boolean receiverTainted = taint(receiver) != null;
            if (outcome && method.equals("matches") && receiverName != null && receiverTainted) {
                found.put(receiverName, Set.of("*"));
            } else if (outcome && (method.equals("equals") || method.equals("equalsIgnoreCase")) && arguments.size() == 1) {
                if (receiverName != null && receiverTainted && constant(arguments.get(0)) != null) {
                    found.put(receiverName, Set.of("*"));
                } else if (constant(receiver) != null && targetName(arguments.get(0)) != null) {
                    found.put(targetName(arguments.get(0)), Set.of("*"));
                }
            } else if (outcome && method.equals("contains") && !receiverTainted && arguments.size() == 1 && targetName(arguments.get(0)) != null) {
                found.put(targetName(arguments.get(0)), Set.of("*"));  // One of a fixed set of allowed values.
            } else if (!outcome && method.equals("contains") && receiverTainted && receiverName != null && arguments.size() == 1
                    && constant(arguments.get(0)) instanceof String text) {
                found.put(receiverName, text.contains("..") || text.contains("/") || text.contains("\\") ? Set.of("path") : Set.of("*"));
            } else if (outcome && arguments.size() >= 1 && targetName(arguments.get(arguments.size() - 1)) != null
                    && (Set.of("isNumeric", "isAlphanumeric", "isAlpha", "isDigits", "isCreatable", "isParsable").contains(method)
                        || String.valueOf(receiver).endsWith("Pattern") && method.equals("matches"))) {
                found.put(targetName(arguments.get(arguments.size() - 1)), Set.of("*"));
            }
            found.keySet().removeIf(name -> !tainted.containsKey(name));
            return found;
        }

        @Override
        public Void visitSwitch(SwitchTree node, Void unused) {
            scan(node.getExpression(), unused);
            Object selector = constant(node.getExpression());
            List<? extends CaseTree> cases = node.getCases();
            int start = selector == null ? -1 : matchingCase(cases, selector);
            if (start < -1) {
                return branch(() -> scan(cases, unused));
            }
            // Only the matching case runs, through to its break.
            for (int index = Math.max(start, 0); start >= 0 && index < cases.size(); index++) {
                List<? extends StatementTree> statements = cases.get(index).getStatements();
                if (statements == null) {
                    scan(cases.get(index).getBody(), unused);
                    break;
                }
                scan(statements, unused);
                if (!statements.isEmpty() && statements.get(statements.size() - 1) instanceof BreakTree) {
                    break;
                }
            }
            return null;
        }

        /** The index of the case a known selector runs, -1 for none, or -2 when it can't be told. */
        private int matchingCase(List<? extends CaseTree> cases, Object selector) {
            int fallback = -1;
            for (int index = 0; index < cases.size(); index++) {
                List<? extends ExpressionTree> labels = cases.get(index).getExpressions();
                if (labels.isEmpty()) {
                    fallback = index;
                }
                for (ExpressionTree label : labels) {
                    Object value = constant(label);
                    if (value == null) {
                        return -2;
                    }
                    if (same(value, selector)) {
                        return index;
                    }
                }
            }
            return fallback;
        }

        private static boolean same(Object left, Object right) {
            if (left instanceof Number a && right instanceof Number b) {
                return a.longValue() == b.longValue();
            }
            if (left instanceof Character a && right instanceof Number b) {
                return a == b.longValue();
            }
            if (left instanceof Number a && right instanceof Character b) {
                return a.longValue() == b;
            }
            return left.equals(right);
        }

        /** A value known before the program runs: literals, constant locals and arithmetic on them. */
        private Object constant(ExpressionTree node) {
            if (node instanceof ParenthesizedTree parenthesized) {
                return constant(parenthesized.getExpression());
            }
            if (node instanceof LiteralTree literal) {
                return literal.getValue();
            }
            if (node instanceof IdentifierTree identifier) {
                return constants.get(identifier.getName().toString());
            }
            if (node instanceof UnaryTree unary && unary.getKind() == Tree.Kind.LOGICAL_COMPLEMENT
                    && constant(unary.getExpression()) instanceof Boolean value) {
                return !value;
            }
            if (node instanceof MethodInvocationTree call && call.getMethodSelect() instanceof MemberSelectTree select
                    && select.getIdentifier().contentEquals("charAt") && call.getArguments().size() == 1
                    && constant(select.getExpression()) instanceof String text
                    && constant(call.getArguments().get(0)) instanceof Number index
                    && index.longValue() >= 0 && index.longValue() < text.length()) {
                return text.charAt((int) index.longValue());
            }
            if (!(node instanceof BinaryTree binary)) {
                return null;
            }
            Object left = constant(binary.getLeftOperand());
            Object right = constant(binary.getRightOperand());
            if (left instanceof Boolean a && right instanceof Boolean b) {
                return switch (binary.getKind()) {
                    case CONDITIONAL_AND -> a && b;
                    case CONDITIONAL_OR -> a || b;
                    case EQUAL_TO -> a.equals(b);
                    case NOT_EQUAL_TO -> !a.equals(b);
                    default -> null;
                };
            }
            if (left instanceof Character c) {
                left = (long) c;
            }
            if (right instanceof Character c) {
                right = (long) c;
            }
            if (!(left instanceof Number a) || !(right instanceof Number b)
                    || left instanceof Double || left instanceof Float || right instanceof Double || right instanceof Float) {
                return null;
            }
            long x = a.longValue();
            long y = b.longValue();
            return switch (binary.getKind()) {
                case PLUS -> x + y;
                case MINUS -> x - y;
                case MULTIPLY -> x * y;
                case DIVIDE -> y == 0 ? null : x / y;
                case REMAINDER -> y == 0 ? null : x % y;
                case LESS_THAN -> x < y;
                case LESS_THAN_EQUAL -> x <= y;
                case GREATER_THAN -> x > y;
                case GREATER_THAN_EQUAL -> x >= y;
                case EQUAL_TO -> x == y;
                case NOT_EQUAL_TO -> x != y;
                default -> null;
            };
        }

        /** A loop may change any local it assigns, so none of them stay known inside it. */
        private void forgetAssignedIn(Tree body) {
            new TreeScanner<Void, Void>() {
                @Override
                public Void visitAssignment(AssignmentTree node, Void unused) {
                    forget(node.getVariable());
                    return super.visitAssignment(node, unused);
                }

                @Override
                public Void visitCompoundAssignment(CompoundAssignmentTree node, Void unused) {
                    forget(node.getVariable());
                    return super.visitCompoundAssignment(node, unused);
                }

                @Override
                public Void visitUnary(UnaryTree node, Void unused) {
                    forget(node.getExpression());
                    return super.visitUnary(node, unused);
                }
            }.scan(body, null);
        }

        private void forget(ExpressionTree target) {
            String name = targetName(target);
            if (name != null) {
                constants.remove(name);
            }
        }

        @Override
        public Void visitUnary(UnaryTree node, Void unused) {
            super.visitUnary(node, unused);
            if (node.getKind() != Tree.Kind.LOGICAL_COMPLEMENT && node.getKind() != Tree.Kind.UNARY_MINUS
                    && node.getKind() != Tree.Kind.UNARY_PLUS && node.getKind() != Tree.Kind.BITWISE_COMPLEMENT) {
                forget(node.getExpression());
            }
            return null;
        }

        @Override
        public Void visitWhileLoop(WhileLoopTree node, Void unused) {
            forgetAssignedIn(node);
            return branch(() -> super.visitWhileLoop(node, unused));
        }

        @Override
        public Void visitDoWhileLoop(DoWhileLoopTree node, Void unused) {
            forgetAssignedIn(node);
            return branch(() -> super.visitDoWhileLoop(node, unused));
        }

        @Override
        public Void visitForLoop(ForLoopTree node, Void unused) {
            forgetAssignedIn(node);
            return branch(() -> super.visitForLoop(node, unused));
        }

        @Override
        public Void visitCase(CaseTree node, Void unused) {
            return branch(() -> super.visitCase(node, unused));
        }

        @Override
        public Void visitCatch(CatchTree node, Void unused) {
            return branch(() -> super.visitCatch(node, unused));
        }

        @Override
        public Void visitConditionalExpression(ConditionalExpressionTree node, Void unused) {
            return branch(() -> super.visitConditionalExpression(node, unused));
        }

        @Override
        public Void visitMethodInvocation(MethodInvocationTree node, Void unused) {
            // A lambda called with what a tainted receiver holds, as in bodyToMono(...).map(body -> ...).
            if (node.getMethodSelect() instanceof MemberSelectTree select) {
                String held = taint(select.getExpression());
                for (ExpressionTree argument : node.getArguments()) {
                    if (held != null && argument instanceof LambdaExpressionTree lambda && !lambda.getParameters().isEmpty()) {
                        tainted.put(lambda.getParameters().get(0).getName().toString(), held);
                    }
                }
                String method = select.getIdentifier().toString();
                if (method.equals("setAttribute") && node.getArguments().size() == 2 && isRequestOrSession(select.getExpression())
                        && constant(node.getArguments().get(0)) instanceof String name) {
                    String value = taint(node.getArguments().get(1));
                    if (value != null) {
                        unit.project.attributes.putIfAbsent(name, baseSource(value));
                    }
                }
            }
            super.visitMethodInvocation(node, unused);
            checkInvocation(node);
            followHelper(node);
            if (trackCollection(node)) {
                return null;
            }
            if (node.getMethodSelect() instanceof MemberSelectTree select
                    && COLLECTING_METHODS.contains(select.getIdentifier().toString())) {
                String source = firstTaint(node.getArguments());
                String name = targetName(rootOf(select.getExpression()));
                if (source != null && name != null) {
                    tainted.put(name, source);
                }
            }
            return null;
        }

        @Override
        public Void visitNewClass(NewClassTree node, Void unused) {
            super.visitNewClass(node, unused);
            checkConstruction(node);
            return null;
        }

        /**
         * Records what an object built from one of the scanned classes keeps from its constructor's
         * arguments, as in this.request = request, so its later method calls read from it.
         */
        private void remember(String variable, ExpressionTree value) {
            fieldInput.remove(variable);
            fieldRequests.remove(variable);
            if (!(value instanceof NewClassTree construction)) {
                return;
            }
            String type = simpleName(construction.getIdentifier());
            ClassInfo info = unit.local.getOrDefault(type, unit.project.classes.get(type));
            MethodTree constructor = info == null ? null : info.method("<init>", construction.getArguments().size());
            if (constructor == null || constructor.getBody() == null) {
                return;
            }
            Map<String, String> inputs = new HashMap<>();
            Set<String> requestParameters = new HashSet<>();
            List<? extends VariableTree> parameters = constructor.getParameters();
            for (int index = 0; index < parameters.size(); index++) {
                ExpressionTree argument = construction.getArguments().get(index);
                String parameter = parameters.get(index).getName().toString();
                String source = taint(argument);
                if (source != null) {
                    inputs.put(parameter, source);
                } else if (requests.contains(String.valueOf(argument))) {
                    requestParameters.add(parameter);
                }
            }
            Map<String, String> fields = new HashMap<>();
            Set<String> requestFields = new HashSet<>();
            for (StatementTree statement : constructor.getBody().getStatements()) {
                if (!(statement instanceof com.sun.source.tree.ExpressionStatementTree expression)
                        || !(expression.getExpression() instanceof AssignmentTree assignment)) {
                    continue;
                }
                String target = String.valueOf(assignment.getVariable());
                String field = target.startsWith("this.") ? target.substring(5) : target;
                String from = String.valueOf(assignment.getExpression());
                if (inputs.containsKey(from)) {
                    fields.put(field, inputs.get(from));
                } else if (requestParameters.contains(from)) {
                    requestFields.add(field);
                }
            }
            if (!fields.isEmpty()) {
                fieldInput.put(variable, fields);
            }
            if (!requestFields.isEmpty()) {
                fieldRequests.put(variable, requestFields);
            }
        }

        /** Follows a tracked list or literal-keyed map through one call; false when it isn't one. */
        private boolean trackCollection(MethodInvocationTree call) {
            if (!(call.getMethodSelect() instanceof MemberSelectTree select) || !(select.getExpression() instanceof IdentifierTree receiver)) {
                return false;
            }
            String name = receiver.getName().toString();
            String method = select.getIdentifier().toString();
            List<? extends ExpressionTree> arguments = call.getArguments();
            List<String> list = lists.get(name);
            if (list != null) {
                Object index = arguments.isEmpty() ? null : constant(arguments.get(0));
                int at = index instanceof Number number ? (int) number.longValue() : -1;
                if (method.equals("add") && arguments.size() == 1) {
                    list.add(taint(arguments.get(0)));
                } else if (method.equals("add") && arguments.size() == 2 && at >= 0 && at <= list.size()) {
                    list.add(at, taint(arguments.get(1)));
                } else if (method.equals("set") && arguments.size() == 2 && at >= 0 && at < list.size()) {
                    list.set(at, taint(arguments.get(1)));
                } else if (method.equals("remove") && arguments.size() == 1 && index instanceof Integer && at < list.size()) {
                    if (at >= 0) {
                        list.remove(at);
                    }
                } else if (method.equals("clear")) {
                    list.clear();
                } else if (Set.of("add", "addAll", "remove", "set", "removeIf", "replaceAll", "sort").contains(method)) {
                    // A change it can't follow by position: the list is treated as one value from now on.
                    lists.remove(name);
                    String source = firstNonNull(list);
                    String added = firstTaint(arguments);
                    bind(name, source != null ? source : added);
                    if (source != null || added != null) {
                        tainted.put(name, source != null ? source : added);
                    }
                }
                return true;
            }
            if (method.equals("put") && arguments.size() == 2 && constant(arguments.get(0)) != null && !tainted.containsKey(name)) {
                keyed.computeIfAbsent(name, key -> new HashMap<>()).put(constant(arguments.get(0)), taint(arguments.get(1)));
                return true;
            }
            if (keyed.containsKey(name) && Set.of("put", "putAll", "putIfAbsent", "merge", "compute").contains(method)) {
                String source = firstNonNull(keyed.remove(name).values());
                String added = firstTaint(arguments);
                if (source != null || added != null) {
                    tainted.put(name, source != null ? source : added);
                }
                return true;
            }
            return false;
        }

        private static String firstNonNull(Iterable<String> values) {
            for (String value : values) {
                if (value != null) {
                    return value;
                }
            }
            return null;
        }

        private boolean isRequestOrSession(ExpressionTree receiver) {
            String text = String.valueOf(receiver);
            return requests.contains(text) || text.endsWith("getSession()") || text.toLowerCase(Locale.ROOT).contains("session");
        }

        /** A value read back from a tracked list by position, or a tracked map by literal key. */
        private String collectionRead(MethodInvocationTree call, MemberSelectTree select) {
            if (!(select.getExpression() instanceof IdentifierTree receiver) || call.getArguments().size() != 1) {
                return null;
            }
            String name = receiver.getName().toString();
            String method = select.getIdentifier().toString();
            Object key = constant(call.getArguments().get(0));
            List<String> list = lists.get(name);
            if (list != null && method.equals("get") && key instanceof Number number) {
                int at = (int) number.longValue();
                return at >= 0 && at < list.size() ? Optional.ofNullable(list.get(at)).orElse("") : null;
            }
            Map<Object, String> values = keyed.get(name);
            if (values != null && (method.equals("get") || method.equals("getOrDefault")) && key != null) {
                return Optional.ofNullable(values.get(key)).orElse("");
            }
            return null;
        }

        private void bind(String name, String source) {
            if (source != null) {
                tainted.put(name, source);
            } else if (nesting == 0) {
                // A value overwritten inside a branch may still be tainted on another path.
                tainted.remove(name);
            }
        }

        private static ExpressionTree rootOf(ExpressionTree expression) {
            ExpressionTree current = expression;
            while (true) {
                if (current instanceof MethodInvocationTree call && call.getMethodSelect() instanceof MemberSelectTree select) {
                    current = select.getExpression();
                } else if (current instanceof MemberSelectTree select) {
                    current = select.getExpression();
                } else if (current instanceof ParenthesizedTree parenthesized) {
                    current = parenthesized.getExpression();
                } else {
                    return current;
                }
            }
        }

        private static String targetName(ExpressionTree target) {
            if (target instanceof IdentifierTree identifier) {
                return identifier.getName().toString();
            }
            if (target instanceof ArrayAccessTree access) {
                return targetName(access.getExpression());
            }
            return null;
        }

        private String taint(ExpressionTree node) {
            if (node == null) {
                return null;
            }
            if (node instanceof IdentifierTree identifier) {
                String name = identifier.getName().toString();
                String source = tainted.get(name);
                if (source == null && lists.containsKey(name)) {
                    source = firstNonNull(lists.get(name));
                }
                if (source == null && keyed.containsKey(name)) {
                    source = firstNonNull(keyed.get(name).values());
                }
                return source;
            }
            if (node instanceof MemberSelectTree select) {
                return taint(select.getExpression());
            }
            if (node instanceof ParenthesizedTree parenthesized) {
                return taint(parenthesized.getExpression());
            }
            if (node instanceof TypeCastTree cast) {
                return taint(cast.getExpression());
            }
            if (node instanceof ArrayAccessTree access) {
                return taint(access.getExpression());
            }
            if (node instanceof BinaryTree binary) {
                if (binary.getKind() != Tree.Kind.PLUS) {
                    return null;
                }
                return combine(taint(binary.getLeftOperand()), taint(binary.getRightOperand()));
            }
            if (node instanceof ConditionalExpressionTree conditional) {
                if (constant(conditional.getCondition()) instanceof Boolean known) {
                    return taint(known ? conditional.getTrueExpression() : conditional.getFalseExpression());
                }
                String first = taint(conditional.getTrueExpression());
                return first != null ? first : taint(conditional.getFalseExpression());
            }
            if (node instanceof NewArrayTree array) {
                return array.getInitializers() == null ? null : firstTaint(array.getInitializers());
            }
            if (node instanceof NewClassTree construction) {
                String type = simpleName(construction.getIdentifier());
                // Text and streams over it carry what they were made from.
                return Set.of("ByteArrayInputStream", "InputSource", "StreamSource", "String", "StringBuffer", "StringBuilder",
                        "StringReader", "URI", "URL").contains(type) ? firstTaint(construction.getArguments()) : null;
            }
            if (node instanceof MethodInvocationTree call) {
                return callTaint(call);
            }
            return null;
        }

        private String firstTaint(List<? extends ExpressionTree> nodes) {
            String found = null;
            for (ExpressionTree node : nodes) {
                found = combine(found, taint(node));
            }
            return found;
        }

        private String callTaint(MethodInvocationTree call) {
            if (!(call.getMethodSelect() instanceof MemberSelectTree select)) {
                return followHelper(call);
            }
            String method = select.getIdentifier().toString();
            String receiver = String.valueOf(select.getExpression());
            String qualified = simpleName(select.getExpression()) + "." + method;
            if (SANITIZERS.contains(qualified)) {
                return null;
            }
            if (KIND_SANITIZERS.containsKey(method)) {
                return safeSource(firstTaint(call.getArguments()), KIND_SANITIZERS.get(method));
            }
            if (PASSTHROUGH_STATIC.contains(qualified)) {
                return firstTaint(call.getArguments());
            }
            if (requests.contains(receiver) && REQUEST_METHODS.contains(method)) {
                boolean named = !call.getArguments().isEmpty() && call.getArguments().get(0) instanceof LiteralTree literal
                        && literal.getValue() instanceof String;
                return receiver + "." + method + "(" + (named ? call.getArguments().get(0) : "") + ")";
            }
            if (NON_TEXT_METHODS.contains(method)) {
                return null;
            }
            String read = collectionRead(call, select);
            if (read != null) {
                return read.isEmpty() ? null : read;
            }
            if (method.equals("getAttribute") && call.getArguments().size() == 1 && isRequestOrSession(select.getExpression())
                    && constant(call.getArguments().get(0)) instanceof String name && unit.project.attributes.containsKey(name)) {
                return receiver + ".getAttribute(\"" + name + "\") (" + unit.project.attributes.get(name) + ")";
            }
            if (!resolve(call).isEmpty()) {
                return followHelper(call);
            }
            String source = taint(select.getExpression());
            if (source != null) {
                return source;
            }
            return ARGUMENT_PASSTHROUGH.contains(method) ? firstTaint(call.getArguments()) : null;
        }

        /**
         * Finds the scanned methods a call may run: in this class, or on a value whose class is known
         * from a field, variable or parameter type, a static call, new, or another call's return type.
         */
        private List<Target> resolve(MethodInvocationTree call) {
            int arguments = call.getArguments().size();
            if (call.getMethodSelect() instanceof IdentifierTree identifier) {
                MethodTree method = owner.method(identifier.getName().toString(), arguments);
                return method == null ? List.of() : List.of(new Target(owner, method));
            }
            if (!(call.getMethodSelect() instanceof MemberSelectTree select)) {
                return List.of();
            }
            String name = select.getIdentifier().toString();
            if (String.valueOf(select.getExpression()).equals("this")) {
                MethodTree method = owner.method(name, arguments);
                return method == null ? List.of() : List.of(new Target(owner, method));
            }
            return unit.project.methods(unit, typeOf(select.getExpression()), name, arguments);
        }

        /** The scanned class a value belongs to, when the source shows it. */
        private String typeOf(ExpressionTree value) {
            ExpressionTree current = value;
            while (current instanceof ParenthesizedTree parenthesized) {
                current = parenthesized.getExpression();
            }
            if (current instanceof NewClassTree construction) {
                return simpleName(construction.getIdentifier());
            }
            if (current instanceof TypeCastTree cast) {
                return simpleName(cast.getType());
            }
            if (current instanceof IdentifierTree identifier) {
                String variable = identifier.getName().toString();
                return types.containsKey(variable) ? types.get(variable)
                        : Character.isUpperCase(variable.charAt(0)) ? variable : null;
            }
            String text = String.valueOf(current);
            if (text.startsWith("this.") && text.indexOf('.', 5) < 0) {
                return owner.fields.get(text.substring(5));
            }
            if (current instanceof MethodInvocationTree call) {
                // A call's result has its method's declared return type.
                for (Target target : resolve(call)) {
                    Tree returns = target.method().getReturnType();
                    if (returns != null && (unit.local.containsKey(simpleName(returns)) || unit.project.classes.containsKey(simpleName(returns)))) {
                        return simpleName(returns);
                    }
                }
            }
            return null;
        }

        /** Walks the scanned methods a call may run, and returns the input their result carries. */
        private String followHelper(MethodInvocationTree call) {
            if (helperReturns.containsKey(call)) {
                return helperReturns.get(call);
            }
            helperReturns.put(call, null);
            String result = null;
            for (Target target : depth >= MAX_HELPER_DEPTH ? List.<Target>of() : resolve(call)) {
                String returned = follow(call, target);
                result = result != null ? result : returned;
            }
            helperReturns.put(call, result);
            return result;
        }

        private String follow(MethodInvocationTree call, Target target) {
            MethodTree helper = target.method();
            String name = helper.getName().toString();
            Map<String, String> passed = new HashMap<>();
            Set<String> passedRequests = new HashSet<>();
            List<? extends VariableTree> parameters = helper.getParameters();
            for (int index = 0; index < parameters.size() && index < call.getArguments().size(); index++) {
                ExpressionTree argument = call.getArguments().get(index);
                String source = taint(argument);
                if (source != null) {
                    passed.put(parameters.get(index).getName().toString(), source);
                } else if (requests.contains(String.valueOf(argument))) {
                    passedRequests.add(parameters.get(index).getName().toString());
                }
            }
            // An object's method also reads what its constructor kept, as this.request or request.
            if (call.getMethodSelect() instanceof MemberSelectTree select && select.getExpression() instanceof IdentifierTree object) {
                String variable = object.getName().toString();
                fieldInput.getOrDefault(variable, Map.of()).forEach((field, source) -> {
                    passed.putIfAbsent(field, source);
                    passed.putIfAbsent("this." + field, source);
                });
                for (String field : fieldRequests.getOrDefault(variable, Set.of())) {
                    passedRequests.add(field);
                    passedRequests.add("this." + field);
                }
            }
            if (passed.isEmpty() && passedRequests.isEmpty()) {
                return null;
            }
            String next = target.owner().unit == unit ? entry : (entry.isEmpty() ? unit.file : entry);
            return UnitAnalyzer.analyzeMethod(target.owner(), helper, passed, passedRequests, label + " -> " + name,
                    depth + 1, active, next);
        }

        private void checkInvocation(MethodInvocationTree call) {
            if (!(call.getMethodSelect() instanceof MemberSelectTree select)) {
                return;
            }
            // A method of one of the scanned classes is followed into, not guessed at by its name.
            if (!resolve(call).isEmpty()) {
                return;
            }
            String method = select.getIdentifier().toString();
            String receiver = String.valueOf(select.getExpression());
            String lower = receiver.toLowerCase(Locale.ROOT);
            String display = receiver + "." + method;
            String qualified = simpleName(select.getExpression()) + "." + method;
            List<? extends ExpressionTree> arguments = call.getArguments();
            String first = arguments.isEmpty() ? null : taint(arguments.get(0));

            if (first != null && SQL_ANY_RECEIVER.contains(method)) {
                unit.report(call, label, "sql", display, first, "high", entry);
            } else if (first != null && SQL_STATEMENT.contains(method) && containsAny(lower, "stmt", "statement", "jdbc", "conn", "template", "db")) {
                unit.report(call, label, "sql", display, first, "high", entry);
            } else if (first != null && SQL_TEMPLATE.contains(method) && containsAny(lower, "jdbc", "template")) {
                unit.report(call, label, "sql", display, first, "high", entry);
            } else if (method.equals("exec") && (lower.contains("runtime") || "Runtime".equals(typeOf(select.getExpression())))
                    && firstTaint(arguments) != null) {
                // Any argument: the command, its arguments or its environment.
                unit.report(call, label, "command", display, firstTaint(arguments), "high", entry);
            } else if (method.equals("command") && "ProcessBuilder".equals(typeOf(select.getExpression())) && firstTaint(arguments) != null) {
                unit.report(call, label, "command", display, firstTaint(arguments), "high", entry);
            } else if (lower.equals("files") || receiver.endsWith(".Files")) {
                String source = firstTaint(arguments);
                if (source != null) {
                    unit.report(call, label, "path", display, source, "medium", entry);
                }
            } else if (qualified.equals("Paths.get") || qualified.equals("Path.of")) {
                String source = firstTaint(arguments);
                if (source != null) {
                    unit.report(call, label, "path", display, source, "medium", entry);
                }
            } else if (first != null && qualified.equals("URI.create")) {
                unit.report(call, label, "ssrf", display, first, "medium", entry);
            } else if (first != null && SSRF_REST_METHODS.contains(method) && lower.contains("rest")) {
                unit.report(call, label, "ssrf", display, first, "medium", entry);
            } else if (first != null && method.equals("uri") && containsAny(lower, "webclient", "client", "builder")) {
                unit.report(call, label, "ssrf", display, first, "medium", entry);
            } else if (first != null && method.equals("lookup") && (containsAny(lower, "ctx", "context", "jndi", "initialcontext")
                    || String.valueOf(typeOf(select.getExpression())).endsWith("Context"))
                    && !startsWithLiteral(arguments.get(0), "java:")) {
                // JNDI lookups of a name from input load remote classes, as Log4Shell did.
                unit.report(call, label, "code", display, first, "high", entry);
            } else if (first != null && (qualified.equals("Class.forName") || qualified.startsWith("MVEL.")
                    || qualified.equals("Ognl.getValue") || qualified.equals("Ognl.parseExpression")
                    || ((method.equals("evaluate") || method.equals("parse") || method.equals("run"))
                        && (lower.contains("groovy") || String.valueOf(typeOf(select.getExpression())).equals("GroovyShell")))
                    || (method.equals("eval") && (lower.contains("elprocessor") || "ELProcessor".equals(typeOf(select.getExpression())))))) {
                unit.report(call, label, "code", display, first, "high", entry);
            } else if ((method.equals("createValueExpression") || method.equals("createMethodExpression")) && arguments.size() >= 2
                    && taint(arguments.get(1)) != null) {
                unit.report(call, label, "code", display, taint(arguments.get(1)), "high", entry);
            } else if (qualified.equals("Velocity.evaluate") && arguments.size() >= 4 && taint(arguments.get(3)) != null) {
                unit.report(call, label, "template", display, taint(arguments.get(3)), "high", entry);
            } else if (first != null && (method.equals("fromXML") && containsAny(lower, "xstream", "xs")
                    || qualified.equals("SerializationUtils.deserialize"))) {
                unit.report(call, label, "deserialization", display, first, "high", entry);
            } else if (first != null && method.equals("eval") && lower.contains("engine")) {
                unit.report(call, label, "code", display, first, "high", entry);
            } else if (first != null && method.equals("parseExpression")) {
                unit.report(call, label, "code", display, first, "high", entry);
            } else if (first != null && method.equals("load") && lower.contains("yaml")) {
                unit.report(call, label, "deserialization", display, first, "high", entry);
            } else if (first != null && qualified.equals("Document.parse")) {
                unit.report(call, label, "nosql", display, first, "medium", entry);
            } else if (RESPONSE_WRITER_METHODS.contains(method) && (receiver.contains("getWriter()") || receiver.contains("getOutputStream()")
                    || RESPONSE_WRITER_TYPES.contains(String.valueOf(typeOf(select.getExpression())))) && firstTaint(arguments) != null) {
                unit.report(call, label, "xss", display, firstTaint(arguments), "medium", entry);
            } else if (first != null && method.equals("sendRedirect")) {
                unit.report(call, label, "redirect", display, first, "medium", entry);
            } else if (method.equals("search") && arguments.size() >= 2 && taint(arguments.get(1)) != null
                    && (containsAny(lower, "ctx", "context", "ldap", "idc", "dir")
                        || String.valueOf(typeOf(select.getExpression())).contains("DirContext"))) {
                unit.report(call, label, "ldap", display, taint(arguments.get(1)), "high", entry);
            } else if (first != null && (method.equals("compile") || method.equals("evaluate"))
                    && ("XPath".equals(typeOf(select.getExpression())) || containsAny(lower, "xpath", "xp"))) {
                unit.report(call, label, "xpath", display, first, "high", entry);
            } else if (first != null && (qualified.equals("Pattern.compile") || qualified.equals("Pattern.matches"))) {
                unit.report(call, label, "redos", display, first, "medium", entry);
            } else if (arguments.size() >= 1 && taint(arguments.get(0)) != null && Set.of("matches", "replaceAll", "replaceFirst", "split").contains(method)
                    && !receiver.isEmpty() && !qualified.startsWith("Pattern.")) {
                // The regular expression itself came from input, as in text.matches(input).
                unit.report(call, label, "redos", display, taint(arguments.get(0)), "medium", entry);
            } else if (first != null && XML_PARSE_METHODS.contains(method)
                    && containsAny(lower, "builder", "parser", "reader", "factory", "unmarshaller", "sax", "xml") && !hardened) {
                unit.report(call, label, "xxe", display, first, "medium", entry);
            }
        }

        private void checkConstruction(NewClassTree node) {
            String type = simpleName(node.getIdentifier());
            String source = firstTaint(node.getArguments());
            if (source == null) {
                return;
            }
            if (type.equals("ProcessBuilder")) {
                unit.report(node, label, "command", "new ProcessBuilder", source, "medium", entry);
            } else if (PATH_CONSTRUCTORS.contains(type)) {
                unit.report(node, label, "path", "new " + type, source, "medium", entry);
            } else if ((type.equals("URL") || type.equals("URI")) && !node.getArguments().isEmpty()
                    && "file".equals(constant(node.getArguments().get(0)))) {
                unit.report(node, label, "path", "new " + type, source, "medium", entry);
            } else if ((type.equals("URL") || type.equals("URI")) && !source.contains(".getRequestUR")) {
                // The request's own URL only names this app, and building a URL fetches nothing.
                unit.report(node, label, "ssrf", "new " + type, source, "medium", entry);
            } else if (type.equals("ObjectInputStream") || type.equals("XMLDecoder")) {
                unit.report(node, label, "deserialization", "new " + type, source, "high", entry);
            } else if (type.equals("BasicQuery")) {
                unit.report(node, label, "nosql", "new BasicQuery", source, "medium", entry);
            }
        }

        /** A name built from a literal prefix, as in "java:comp/env/" + name, which stays in the local namespace. */
        private static boolean startsWithLiteral(ExpressionTree value, String prefix) {
            ExpressionTree current = value;
            while (current instanceof BinaryTree binary && binary.getKind() == Tree.Kind.PLUS) {
                current = binary.getLeftOperand();
            }
            return current instanceof LiteralTree literal && literal.getValue() instanceof String text && text.startsWith(prefix);
        }

        private static boolean containsAny(String value, String... parts) {
            for (String part : parts) {
                if (value.contains(part)) {
                    return true;
                }
            }
            return false;
        }
    }
}
