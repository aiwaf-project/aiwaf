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
import com.sun.source.tree.LineMap;
import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.MemberSelectTree;
import com.sun.source.tree.MethodInvocationTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.NewArrayTree;
import com.sun.source.tree.NewClassTree;
import com.sun.source.tree.ParenthesizedTree;
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
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
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
 * handling. A clean report doesn't prove code is safe, because input that crosses classes or
 * dynamic calls can be missed. This needs the JDK's compiler module, so a plain JRE can't run it.
 */
public final class JavaSourceAnalysisCore {
    public static final String SCHEMA_VERSION = "1.0";

    private static final Set<String> SKIPPED_DIRECTORIES = Set.of(".git", ".gradle", ".idea", "build", "node_modules", "out", "target");
    private static final long MAX_FILE_BYTES = 2L * 1024L * 1024L;
    private static final int MAX_HELPER_DEPTH = 3;

    private static final Set<String> INPUT_ANNOTATIONS = Set.of(
            "CookieValue", "CookieParam", "FormParam", "HeaderParam", "MatrixParam", "MatrixVariable", "ModelAttribute",
            "PathParam", "PathVariable", "QueryParam", "RequestBody", "RequestHeader", "RequestParam", "RequestPart");
    private static final Set<String> WHOLE_BODY_ANNOTATIONS = Set.of("ModelAttribute", "RequestBody", "RequestPart");
    private static final Set<String> REQUEST_TYPES = Set.of(
            "HttpServletRequest", "NativeWebRequest", "ServletRequest", "ServerHttpRequest", "WebRequest");
    private static final Set<String> REQUEST_METHODS = Set.of(
            "getCookies", "getHeader", "getHeaders", "getInputStream", "getParameter", "getParameterMap", "getParameterNames",
            "getParameterValues", "getPart", "getParts", "getPathInfo", "getQueryString", "getReader", "getRequestURI", "getRequestURL");
    private static final Set<String> SANITIZERS = Set.of(
            "Boolean.parseBoolean", "Boolean.valueOf", "Double.parseDouble", "Double.valueOf", "FilenameUtils.getName",
            "HtmlUtils.htmlEscape", "Integer.parseInt", "Integer.valueOf", "Long.parseLong", "Long.valueOf", "UUID.fromString");
    private static final Set<String> PASSTHROUGH_STATIC = Set.of(
            "Arrays.asList", "List.of", "MessageFormat.format", "Objects.toString", "Optional.of", "Optional.ofNullable",
            "Set.of", "String.format", "String.join", "String.valueOf", "URLDecoder.decode");
    private static final Set<String> ARGUMENT_PASSTHROUGH = Set.of("append", "concat", "formatted", "insert", "join", "replace", "replaceAll", "replaceFirst", "resolve");
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

    public record Finding(String file, long line, String function, String sink, String call, String source, String confidence) {}

    public record Route(String path, List<String> methods, String view, String name, ApiDetection api, Map<String, Object> auth) {}

    public record Report(int filesScanned, List<Finding> findings, List<Map<String, String>> errors, List<Route> routes) {
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
                items.add(item);
            }
            out.put("findings", items);
            out.put("errors", errors);
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
            for (CompilationUnitTree unit : trees) {
                String display = Path.of(unit.getSourceFile().toUri()).toString().replace('\\', '/');
                if (failed.contains(display)) {
                    continue;
                }
                parsed++;
                new UnitAnalyzer(unit, display, positions, findings, routes).run();
            }
        }
        findings.sort(Comparator.comparing(Finding::file).thenComparingLong(Finding::line).thenComparing(Finding::sink));
        routes.sort(Comparator.comparing(Route::path));
        return new Report(parsed, List.copyOf(findings), List.copyOf(errors), List.copyOf(routes));
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
        private final Set<String> seen = new HashSet<>();

        UnitAnalyzer(CompilationUnitTree unit, String file, SourcePositions positions, List<Finding> findings, List<Route> routes) {
            this.unit = unit;
            this.file = file;
            this.positions = positions;
            this.findings = findings;
            this.routes = routes;
        }

        void run() {
            new TreeScanner<Void, Void>() {
                @Override
                public Void visitClass(ClassTree node, Void unused) {
                    analyzeClass(node);
                    return super.visitClass(node, unused);
                }
            }.scan(unit, null);
        }

        private void analyzeClass(ClassTree type) {
            Map<String, MethodTree> methods = new HashMap<>();
            for (Tree member : type.getMembers()) {
                if (member instanceof MethodTree method) {
                    methods.putIfAbsent(method.getName().toString(), method);
                }
            }
            String className = type.getSimpleName().toString();
            for (Tree member : type.getMembers()) {
                if (!(member instanceof MethodTree method) || method.getBody() == null) {
                    continue;
                }
                Map<String, String> tainted = new HashMap<>();
                Set<String> requests = new HashSet<>();
                for (VariableTree parameter : method.getParameters()) {
                    AnnotationTree annotation = parameter.getModifiers().getAnnotations().stream()
                            .filter(item -> INPUT_ANNOTATIONS.contains(simpleName(item.getAnnotationType())))
                            .findFirst().orElse(null);
                    String name = parameter.getName().toString();
                    if (annotation != null) {
                        tainted.put(name, inputSource(annotation, name));
                    } else if (REQUEST_TYPES.contains(simpleName(parameter.getType()))) {
                        requests.add(name);
                    }
                }
                if (!tainted.isEmpty() || !requests.isEmpty()) {
                    analyzeMethod(method, tainted, requests, className + "." + method.getName(), methods, 0, new HashSet<>());
                }
            }
            collectRoutes(type, className);
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

        private void analyzeMethod(MethodTree method, Map<String, String> tainted, Set<String> requests, String label,
                                   Map<String, MethodTree> methods, int depth, Set<String> active) {
            String key = method.getName() + ":" + method.getParameters().size();
            if (depth > MAX_HELPER_DEPTH || method.getBody() == null || !active.add(key)) {
                return;
            }
            new MethodAnalyzer(this, tainted, requests, label, methods, depth, active).scan(method.getBody().getStatements(), null);
            active.remove(key);
        }

        void report(Tree node, String function, String sink, String call, String source, String confidence) {
            long position = positions.getStartPosition(unit, node);
            LineMap lines = unit.getLineMap();
            long line = position >= 0 ? lines.getLineNumber(position) : 0;
            if (seen.add(line + ":" + sink + ":" + call)) {
                findings.add(new Finding(file, line, function, sink, call, source, confidence));
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
        private final UnitAnalyzer unit;
        private final Map<String, String> tainted;
        private final Set<String> requests;
        private final String label;
        private final Map<String, MethodTree> methods;
        private final int depth;
        private final Set<String> active;
        private int nesting;

        MethodAnalyzer(UnitAnalyzer unit, Map<String, String> tainted, Set<String> requests, String label,
                       Map<String, MethodTree> methods, int depth, Set<String> active) {
            this.unit = unit;
            this.tainted = new HashMap<>(tainted);
            this.requests = requests;
            this.label = label;
            this.methods = methods;
            this.depth = depth;
            this.active = active;
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
            if (node.getInitializer() != null) {
                bind(node.getName().toString(), taint(node.getInitializer()));
            }
            return null;
        }

        @Override
        public Void visitAssignment(AssignmentTree node, Void unused) {
            super.visitAssignment(node, unused);
            String name = targetName(node.getVariable());
            if (name != null) {
                bind(name, taint(node.getExpression()));
            }
            return null;
        }

        @Override
        public Void visitCompoundAssignment(CompoundAssignmentTree node, Void unused) {
            super.visitCompoundAssignment(node, unused);
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
            return super.visitEnhancedForLoop(node, unused);
        }

        @Override
        public Void visitMethodInvocation(MethodInvocationTree node, Void unused) {
            super.visitMethodInvocation(node, unused);
            checkInvocation(node);
            followHelper(node);
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
                return tainted.get(identifier.getName().toString());
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
                String left = taint(binary.getLeftOperand());
                return left != null ? left : taint(binary.getRightOperand());
            }
            if (node instanceof ConditionalExpressionTree conditional) {
                String first = taint(conditional.getTrueExpression());
                return first != null ? first : taint(conditional.getFalseExpression());
            }
            if (node instanceof NewArrayTree array) {
                return array.getInitializers() == null ? null : firstTaint(array.getInitializers());
            }
            if (node instanceof NewClassTree construction) {
                String type = simpleName(construction.getIdentifier());
                return Set.of("String", "StringBuffer", "StringBuilder").contains(type) ? firstTaint(construction.getArguments()) : null;
            }
            if (node instanceof MethodInvocationTree call) {
                return callTaint(call);
            }
            return null;
        }

        private String firstTaint(List<? extends ExpressionTree> nodes) {
            for (ExpressionTree node : nodes) {
                String source = taint(node);
                if (source != null) {
                    return source;
                }
            }
            return null;
        }

        private String callTaint(MethodInvocationTree call) {
            if (!(call.getMethodSelect() instanceof MemberSelectTree select)) {
                return null;
            }
            String method = select.getIdentifier().toString();
            String receiver = String.valueOf(select.getExpression());
            String qualified = simpleName(select.getExpression()) + "." + method;
            if (SANITIZERS.contains(qualified)) {
                return null;
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
            String source = taint(select.getExpression());
            if (source != null) {
                return source;
            }
            return ARGUMENT_PASSTHROUGH.contains(method) ? firstTaint(call.getArguments()) : null;
        }

        private void followHelper(MethodInvocationTree call) {
            String name;
            if (call.getMethodSelect() instanceof IdentifierTree identifier) {
                name = identifier.getName().toString();
            } else if (call.getMethodSelect() instanceof MemberSelectTree select && "this".equals(String.valueOf(select.getExpression()))) {
                name = select.getIdentifier().toString();
            } else {
                return;
            }
            MethodTree helper = methods.get(name);
            if (helper == null || depth >= MAX_HELPER_DEPTH) {
                return;
            }
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
            if (!passed.isEmpty() || !passedRequests.isEmpty()) {
                unit.analyzeMethod(helper, passed, passedRequests, label + " -> " + name, methods, depth + 1, active);
            }
        }

        private void checkInvocation(MethodInvocationTree call) {
            if (!(call.getMethodSelect() instanceof MemberSelectTree select)) {
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
                unit.report(call, label, "sql", display, first, "high");
            } else if (first != null && SQL_STATEMENT.contains(method) && containsAny(lower, "stmt", "statement", "jdbc", "conn", "template", "db")) {
                unit.report(call, label, "sql", display, first, "high");
            } else if (first != null && SQL_TEMPLATE.contains(method) && containsAny(lower, "jdbc", "template")) {
                unit.report(call, label, "sql", display, first, "high");
            } else if (first != null && method.equals("exec") && lower.contains("runtime")) {
                unit.report(call, label, "command", display, first, "high");
            } else if (lower.equals("files") || receiver.endsWith(".Files")) {
                String source = firstTaint(arguments);
                if (source != null) {
                    unit.report(call, label, "path", display, source, "medium");
                }
            } else if (qualified.equals("Paths.get") || qualified.equals("Path.of")) {
                String source = firstTaint(arguments);
                if (source != null) {
                    unit.report(call, label, "path", display, source, "medium");
                }
            } else if (first != null && qualified.equals("URI.create")) {
                unit.report(call, label, "ssrf", display, first, "medium");
            } else if (first != null && SSRF_REST_METHODS.contains(method) && lower.contains("rest")) {
                unit.report(call, label, "ssrf", display, first, "medium");
            } else if (first != null && method.equals("uri") && containsAny(lower, "webclient", "client", "builder")) {
                unit.report(call, label, "ssrf", display, first, "medium");
            } else if (first != null && method.equals("eval") && lower.contains("engine")) {
                unit.report(call, label, "code", display, first, "high");
            } else if (first != null && method.equals("parseExpression")) {
                unit.report(call, label, "code", display, first, "high");
            } else if (first != null && method.equals("load") && lower.contains("yaml")) {
                unit.report(call, label, "deserialization", display, first, "high");
            } else if (first != null && qualified.equals("Document.parse")) {
                unit.report(call, label, "nosql", display, first, "medium");
            }
        }

        private void checkConstruction(NewClassTree node) {
            String type = simpleName(node.getIdentifier());
            String source = firstTaint(node.getArguments());
            if (source == null) {
                return;
            }
            if (type.equals("ProcessBuilder")) {
                unit.report(node, label, "command", "new ProcessBuilder", source, "medium");
            } else if (PATH_CONSTRUCTORS.contains(type)) {
                unit.report(node, label, "path", "new " + type, source, "medium");
            } else if (type.equals("URL") || type.equals("URI")) {
                unit.report(node, label, "ssrf", "new " + type, source, "medium");
            } else if (type.equals("ObjectInputStream") || type.equals("XMLDecoder")) {
                unit.report(node, label, "deserialization", "new " + type, source, "high");
            } else if (type.equals("BasicQuery")) {
                unit.report(node, label, "nosql", "new BasicQuery", source, "medium");
            }
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
