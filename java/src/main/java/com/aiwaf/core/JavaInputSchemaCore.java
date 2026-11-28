package com.aiwaf.core;

import com.sun.source.tree.AnnotationTree;
import com.sun.source.tree.AssignmentTree;
import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.ClassTree;
import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.ExpressionTree;
import com.sun.source.tree.IdentifierTree;
import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.MemberSelectTree;
import com.sun.source.tree.MethodInvocationTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.NewArrayTree;
import com.sun.source.tree.ParameterizedTypeTree;
import com.sun.source.tree.ArrayTypeTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.VariableTree;
import com.sun.source.util.SourcePositions;
import com.sun.source.util.TreePath;
import com.sun.source.util.TreePathScanner;
import com.sun.source.util.TreeScanner;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * What each Spring or servlet handler reads from the query and body, with types, for the route
 * manifest. {@code init} saves these as each route's {@code inputs} (name to int, float, bool, uuid
 * or text) and {@code inputs_complete}: true only when every read was understood. Requests are
 * then reviewed against them: see {@link RouteSchemaCore}.
 *
 * <p>Anything not understood makes the list incomplete rather than guessed, and says why: the
 * request passed to a method that isn't followed ({@code helper}), the parameter map read or a
 * {@code Map} bound ({@code container_whole}), a key that isn't a constant ({@code dynamic_key}),
 * the raw body, query string or a multipart part read ({@code raw_body}), a body type that isn't in
 * the sources ({@code external_class}), a class with {@code @JsonAnySetter} ({@code model_fields}),
 * or {@code @InitBinder} and {@code @ModelAttribute} methods ({@code class_view}). Body classes
 * come from every scanned file, and the request is followed into static methods of other classes.
 */
public final class JavaInputSchemaCore {
    /** Why a handler's list is incomplete: a code from the shared list, a detail, and its line (0 when unknown). */
    public record Reason(String code, String detail, long line) {}

    /** One handler's inputs, labelled Class.method as the audit labels its findings. */
    /** nested: the keys inside object-valued inputs, from a body class's nested classes. */
    public record Inputs(String function, String file, Map<String, String> inputs, boolean complete, List<Reason> reasons,
                         Map<String, List<String>> nested) {
        public Inputs(String function, String file, Map<String, String> inputs, boolean complete) {
            this(function, file, inputs, complete, List.of());
        }

        public Inputs(String function, String file, Map<String, String> inputs, boolean complete, List<Reason> reasons) {
            this(function, file, inputs, complete, reasons, Map.of());
        }
    }

    /** A reason as the manifest and reports save it: code, detail and, when known, line. */
    public static Map<String, Object> asMap(Reason reason) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("code", reason.code());
        out.put("detail", reason.detail());
        if (reason.line() > 0) out.put("line", reason.line());
        return out;
    }

    /** The reasons one handler's list is incomplete, with lines from the unit they're in. */
    private static final class Gaps {
        final List<Reason> reasons = new ArrayList<>();
        // Keys inside inputs a body class parsed from the raw body gives (readValue(..., Dto.class)).
        final Map<String, List<String>> nested = new java.util.TreeMap<>();
        final SourcePositions positions;

        Gaps(SourcePositions positions) {
            this.positions = positions;
        }

        void add(CompilationUnitTree unit, String code, String detail, Tree at) {
            long line = 0;
            if (positions != null && unit != null && at != null) {
                long start = positions.getStartPosition(unit, at);
                if (start >= 0) line = unit.getLineMap().getLineNumber(start);
            }
            Reason reason = new Reason(code, detail, line);
            if (!reasons.contains(reason)) reasons.add(reason);
        }
    }

    private static final Set<String> MAPPINGS = Set.of("DeleteMapping", "GetMapping", "PatchMapping", "PostMapping", "PutMapping", "RequestMapping");
    private static final Set<String> SERVLET_METHODS = Set.of("doGet", "doPost", "doPut", "doDelete", "doPatch", "service");
    private static final Set<String> REQUEST_TYPES = Set.of("HttpServletRequest", "ServletRequest");
    // Parameters that carry no query or body values.
    private static final Set<String> NOT_INPUTS = Set.of(
            "HttpServletResponse", "ServletResponse", "Model", "ModelMap", "BindingResult", "Errors", "Principal", "Authentication",
            "Locale", "HttpSession", "SessionStatus", "RedirectAttributes", "UriComponentsBuilder", "HttpHeaders", "TimeZone",
            "ZoneId", "HttpMethod");
    private static final Set<String> NOT_INPUT_ANNOTATIONS = Set.of(
            "RequestHeader", "CookieValue", "AuthenticationPrincipal", "RequestAttribute", "SessionAttribute", "Value", "PathVariable");
    private static final Map<String, String> TYPES = Map.ofEntries(
            Map.entry("int", "int"), Map.entry("Integer", "int"), Map.entry("long", "int"), Map.entry("Long", "int"),
            Map.entry("short", "int"), Map.entry("Short", "int"), Map.entry("byte", "int"), Map.entry("Byte", "int"),
            Map.entry("BigInteger", "int"), Map.entry("double", "float"), Map.entry("Double", "float"), Map.entry("float", "float"),
            Map.entry("Float", "float"), Map.entry("BigDecimal", "float"), Map.entry("boolean", "bool"), Map.entry("Boolean", "bool"),
            Map.entry("UUID", "uuid"), Map.entry("String", "text"), Map.entry("CharSequence", "text"), Map.entry("char", "text"),
            Map.entry("Character", "text"), Map.entry("LocalDate", "text"), Map.entry("LocalDateTime", "text"),
            Map.entry("Instant", "text"), Map.entry("OffsetDateTime", "text"), Map.entry("ZonedDateTime", "text"));
    private static final Map<String, String> PARSERS = Map.of(
            "Integer.parseInt", "int", "Integer.valueOf", "int", "Long.parseLong", "int", "Long.valueOf", "int",
            "Double.parseDouble", "float", "Double.valueOf", "float", "Float.parseFloat", "float",
            "Boolean.parseBoolean", "bool", "Boolean.valueOf", "bool", "UUID.fromString", "uuid");
    private static final Set<String> CONTAINERS = Set.of("List", "Set", "Collection", "Optional", "Iterable");
    // How deep nested keys are kept, as the other languages' init does.
    private static final int MAX_NESTED_DEPTH = 4;
    // Request methods that read the query or body other than one named parameter at a time.
    private static final Set<String> RAW_READS = Set.of(
            "getParameterMap", "getParameterNames", "getInputStream", "getReader", "getQueryString", "getParts", "getPart",
            "getRequestURL", "getAttribute");

    private JavaInputSchemaCore() {}

    private static String simpleName(Tree tree) {
        if (tree == null) return "";
        if (tree instanceof ParameterizedTypeTree parameterized) return simpleName(parameterized.getType());
        if (tree instanceof AnnotationTree annotation) return simpleName(annotation.getAnnotationType());
        String text = tree.toString();
        int generic = text.indexOf('<');
        if (generic >= 0) text = text.substring(0, generic);
        return text.substring(text.lastIndexOf('.') + 1).trim();
    }

    private static Map<String, AnnotationTree> annotations(List<? extends AnnotationTree> items) {
        Map<String, AnnotationTree> out = new HashMap<>();
        for (AnnotationTree item : items) out.put(simpleName(item.getAnnotationType()), item);
        return out;
    }

    /** The first string in an annotation's value, name or path, as in @RequestParam("q"). */
    private static String annotationName(AnnotationTree annotation) {
        for (ExpressionTree argument : annotation.getArguments()) {
            ExpressionTree value = argument;
            if (argument instanceof AssignmentTree assignment) {
                String key = assignment.getVariable().toString();
                if (!key.equals("value") && !key.equals("name")) continue;
                value = assignment.getExpression();
            }
            if (value instanceof NewArrayTree array && array.getInitializers() != null && !array.getInitializers().isEmpty()) {
                value = array.getInitializers().get(0);
            }
            if (value instanceof LiteralTree literal && literal.getValue() instanceof String text && !text.isEmpty()) return text;
        }
        return null;
    }

    private static void merge(Map<String, String> inputs, String name, String kind) {
        // A name read as two types, or as text anywhere, is text: the weaker reading wins.
        inputs.merge(name, kind, (old, now) -> old.equals(now) ? old : "text");
    }

    /** int, float, bool, uuid or text for a type, through List, Set, Optional and arrays; null for others. */
    private static String scalar(Tree type) {
        if (type instanceof ArrayTypeTree array) return scalar(array.getType());
        if (type instanceof ParameterizedTypeTree parameterized && CONTAINERS.contains(simpleName(parameterized.getType()))
                && parameterized.getTypeArguments().size() == 1) {
            return scalar(parameterized.getTypeArguments().get(0));
        }
        return TYPES.get(simpleName(type));
    }

    /** Classes and records in the project whose fields a body or form binds to. */
    private static final class Dtos {
        final Map<String, ClassTree> classes = new HashMap<>();
        final Map<ClassTree, CompilationUnitTree> units = new HashMap<>();
        final Set<String> enums = new java.util.HashSet<>();

        Dtos(List<CompilationUnitTree> units) {
            for (CompilationUnitTree unit : units) {
                new TreeScanner<Void, Void>() {
                    @Override
                    public Void visitClass(ClassTree node, Void unused) {
                        classes.putIfAbsent(node.getSimpleName().toString(), node);
                        Dtos.this.units.putIfAbsent(node, unit);
                        if (node.getKind() == Tree.Kind.ENUM) enums.add(node.getSimpleName().toString());
                        return super.visitClass(node, unused);
                    }
                }.scan(unit, null);
            }
        }

        /** A class's bound fields and their types, or null when they can't all be known. */
        Map<String, String> fields(Tree type, int depth) {
            ClassTree node = classes.get(simpleName(type));
            if (node == null || depth > 8 || node.getKind() == Tree.Kind.ENUM || node.getKind() == Tree.Kind.INTERFACE) return null;
            Map<String, String> fields = new LinkedHashMap<>();
            if (node.getExtendsClause() != null) {
                Map<String, String> inherited = fields(node.getExtendsClause(), depth + 1);
                if (inherited == null) return null;
                fields.putAll(inherited);
            }
            for (Tree member : node.getMembers()) {
                if (member instanceof MethodTree method && annotations(method.getModifiers().getAnnotations()).containsKey("JsonAnySetter")) {
                    return null;  // Takes any property sent.
                }
                if (!(member instanceof VariableTree field) || field.getModifiers().getFlags().contains(javax.lang.model.element.Modifier.STATIC)) {
                    continue;
                }
                Map<String, AnnotationTree> marks = annotations(field.getModifiers().getAnnotations());
                if (marks.containsKey("JsonIgnore")) continue;
                if (marks.containsKey("JsonAnySetter")) return null;
                String name = marks.containsKey("JsonProperty") && annotationName(marks.get("JsonProperty")) != null
                        ? annotationName(marks.get("JsonProperty")) : field.getName().toString();
                String kind = scalar(field.getType());
                // An enum, a nested object or a map: text, which fits any value.
                merge(fields, name, kind == null ? "text" : kind);
            }
            return fields;
        }

        /**
         * The keys inside a body class's fields whose type is another class of the scanned sources,
         * by path: what Jackson can fill in it, as {"author": ["name"]}; for a list or array of such
         * classes each element's, as {"lines[]": ["sku"]}; and deeper, as "lines[].product".
         */
        Map<String, List<String>> nested(Tree type) {
            Map<String, List<String>> nested = new java.util.TreeMap<>();
            nestedPaths(type, "", 0, nested, new java.util.HashSet<>());
            return nested;
        }

        private void nestedPaths(Tree type, String prefix, int depth, Map<String, List<String>> out, Set<String> seen) {
            String className = simpleName(type);
            ClassTree node = classes.get(className);
            if (node == null || node.getKind() == Tree.Kind.ENUM || node.getKind() == Tree.Kind.INTERFACE
                    || depth >= MAX_NESTED_DEPTH || !seen.add(className)) return;
            for (Tree member : node.getMembers()) {
                if (!(member instanceof VariableTree field) || field.getModifiers().getFlags().contains(javax.lang.model.element.Modifier.STATIC)) continue;
                Map<String, AnnotationTree> marks = annotations(field.getModifiers().getAnnotations());
                Tree element = field.getType();
                boolean list = false;
                // Through Optional, and into a List, Set or array's element type.
                while (true) {
                    if (element instanceof ArrayTypeTree array) {
                        element = array.getType();
                        list = true;
                    } else if (element instanceof ParameterizedTypeTree parameterized && CONTAINERS.contains(simpleName(parameterized.getType()))
                            && parameterized.getTypeArguments().size() == 1) {
                        list = list || !"Optional".equals(simpleName(parameterized.getType()));
                        element = parameterized.getTypeArguments().get(0);
                    } else {
                        break;
                    }
                }
                if (marks.containsKey("JsonIgnore") || TYPES.containsKey(simpleName(element)) || isEnum(element)) continue;
                Map<String, String> inner = fields(element, 1);
                if (inner == null) continue;
                String name = marks.containsKey("JsonProperty") && annotationName(marks.get("JsonProperty")) != null
                        ? annotationName(marks.get("JsonProperty")) : field.getName().toString();
                String path = prefix + name + (list ? "[]" : "");
                out.put(path, List.copyOf(new java.util.TreeSet<>(inner.keySet())));
                nestedPaths(element, path + ".", depth + 1, out, seen);
            }
            seen.remove(className);
        }

        boolean isEnum(Tree type) {
            return enums.contains(simpleName(type));
        }

        /** Why a class's fields can't be read: (code, detail). */
        String[] why(Tree type) {
            String name = simpleName(type);
            ClassTree node = classes.get(name);
            if (node == null) return new String[] {"external_class", name + " isn't in the scanned sources"};
            if (node.getKind() == Tree.Kind.INTERFACE || node.getKind() == Tree.Kind.ENUM) {
                return new String[] {"external_class", name + " is an interface or enum"};
            }
            if (node.getExtendsClause() != null && fields(node.getExtendsClause(), 1) == null) return why(node.getExtendsClause());
            return new String[] {"model_fields", name + " takes any property (@JsonAnySetter)"};
        }
    }

    /** Each handler's inputs in the parsed units, with each unit's display name. */
    public static List<Inputs> inputs(List<CompilationUnitTree> units, java.util.function.Function<CompilationUnitTree, String> display) {
        return inputs(units, display, null);
    }

    /** Each handler's inputs; with positions, each reason carries its line. */
    public static List<Inputs> inputs(List<CompilationUnitTree> units, java.util.function.Function<CompilationUnitTree, String> display,
                                      SourcePositions positions) {
        Dtos dtos = new Dtos(units);
        List<Inputs> results = new ArrayList<>();
        for (CompilationUnitTree unit : units) {
            String file = display.apply(unit);
            new TreeScanner<Void, Void>() {
                @Override
                public Void visitClass(ClassTree type, Void unused) {
                    handlers(unit, type, file, dtos, results, positions);
                    return super.visitClass(type, unused);
                }
            }.scan(unit, null);
        }
        return results;
    }

    private static void handlers(CompilationUnitTree unit, ClassTree type, String file, Dtos dtos, List<Inputs> results,
                                 SourcePositions positions) {
        // Methods every handler of the class runs first, which may read anything.
        MethodTree shared = type.getMembers().stream().filter(member -> member instanceof MethodTree method
                && (annotations(method.getModifiers().getAnnotations()).containsKey("InitBinder")
                    || annotations(method.getModifiers().getAnnotations()).containsKey("ModelAttribute")))
                .map(MethodTree.class::cast).findFirst().orElse(null);
        for (Tree member : type.getMembers()) {
            if (!(member instanceof MethodTree method) || method.getBody() == null) continue;
            Map<String, AnnotationTree> marks = annotations(method.getModifiers().getAnnotations());
            boolean mapped = MAPPINGS.stream().anyMatch(marks::containsKey);
            boolean servlet = SERVLET_METHODS.contains(method.getName().toString())
                    && method.getParameters().stream().anyMatch(parameter -> REQUEST_TYPES.contains(simpleName(parameter.getType())));
            if (!mapped && !servlet) continue;
            Map<String, String> inputs = new LinkedHashMap<>();
            Gaps gaps = new Gaps(positions);
            if (shared != null) {
                gaps.add(unit, "class_view", type.getSimpleName() + "." + shared.getName()
                        + " runs before every handler and may read anything", shared);
            }
            List<String> requests = new ArrayList<>();
            Map<String, List<String>> nested = new java.util.TreeMap<>();
            for (VariableTree parameter : method.getParameters()) {
                parameter(unit, parameter, dtos, inputs, gaps, requests);
                if (annotations(parameter.getModifiers().getAnnotations()).containsKey("RequestBody")) nested.putAll(dtos.nested(parameter.getType()));
            }
            if (!requests.isEmpty()) reads(unit, type, method, requests, inputs, gaps, dtos, new java.util.HashSet<>());
            nested.putAll(gaps.nested);
            results.add(new Inputs(type.getSimpleName() + "." + method.getName(), file, inputs, gaps.reasons.isEmpty(),
                    List.copyOf(gaps.reasons), Map.copyOf(nested)));
        }
    }

    private static void parameter(CompilationUnitTree unit, VariableTree parameter, Dtos dtos, Map<String, String> inputs, Gaps gaps,
                                  List<String> requests) {
        Map<String, AnnotationTree> marks = annotations(parameter.getModifiers().getAnnotations());
        String type = simpleName(parameter.getType());
        String name = parameter.getName().toString();
        if (REQUEST_TYPES.contains(type)) {
            requests.add(name);
            return;
        }
        if (NOT_INPUTS.contains(type) || marks.keySet().stream().anyMatch(NOT_INPUT_ANNOTATIONS::contains)) return;
        if (type.equals("Pageable")) {
            merge(inputs, "page", "int");
            merge(inputs, "size", "int");
            merge(inputs, "sort", "text");
            return;
        }
        if (Set.of("WebRequest", "NativeWebRequest").contains(type)) {
            gaps.add(unit, "helper", "parameter " + name + " is the whole request (" + type + ")", parameter);
            return;
        }
        if (Set.of("MultipartFile", "MultipartHttpServletRequest", "HttpEntity", "RequestEntity", "InputStream", "Reader").contains(type)
                || marks.containsKey("RequestPart")) {
            gaps.add(unit, "raw_body", "parameter " + name + " reads the raw body or a multipart part (" + type + ")", parameter);
            return;
        }
        String kind = scalar(parameter.getType());
        if (marks.containsKey("RequestParam") || (kind != null && !marks.containsKey("RequestBody") && !marks.containsKey("ModelAttribute"))) {
            if (kind == null && !dtos.isEnum(parameter.getType())) {
                // A map of every parameter, or a type that isn't known.
                if (Set.of("Map", "MultiValueMap", "HashMap", "LinkedHashMap").contains(type)) {
                    gaps.add(unit, "container_whole", "parameter " + name + " binds every parameter (" + type + ")", parameter);
                } else {
                    gaps.add(unit, "external_class", "parameter " + name + " has a type that isn't known (" + type + ")", parameter);
                }
                return;
            }
            String named = marks.containsKey("RequestParam") ? annotationName(marks.get("RequestParam")) : null;
            merge(inputs, named != null ? named : name, kind == null ? "text" : kind);
            return;
        }
        // A @RequestBody, or a @ModelAttribute or other object bound from the form and query.
        Map<String, String> fields = dtos.fields(parameter.getType(), 0);
        if (fields == null) {
            if (Set.of("Map", "MultiValueMap", "HashMap", "LinkedHashMap", "JsonNode", "ObjectNode", "Object", "String").contains(type)) {
                gaps.add(unit, "container_whole", "parameter " + name + " takes the whole body (" + type + ")", parameter);
            } else {
                String[] why = dtos.why(parameter.getType());
                gaps.add(unit, why[0], why[1], parameter);
            }
            return;
        }
        fields.forEach((field, fieldKind) -> merge(inputs, field, fieldKind));
    }

    /** The class's static final String constants, as private static final String PAGE = "page". */
    private static Map<String, String> constants(ClassTree type) {
        Map<String, String> out = new HashMap<>();
        for (Tree member : type.getMembers()) {
            if (member instanceof VariableTree field && field.getInitializer() instanceof LiteralTree literal
                    && literal.getValue() instanceof String text
                    && field.getModifiers().getFlags().containsAll(Set.of(javax.lang.model.element.Modifier.STATIC, javax.lang.model.element.Modifier.FINAL))) {
                out.put(field.getName().toString(), text);
            }
        }
        return out;
    }

    /**
     * The method a call names, as helper(request) or this.helper(request) in the class, or
     * Helpers.helper(request), a static method of another scanned class, with as many parameters.
     */
    private static Object[] method(ClassTree type, CompilationUnitTree unit, MethodInvocationTree call, Dtos dtos) {
        ExpressionTree select = call.getMethodSelect();
        ClassTree owner = type;
        String name = null;
        if (select instanceof IdentifierTree identifier) name = identifier.getName().toString();
        else if (select instanceof MemberSelectTree member) {
            String target = member.getExpression().toString();
            if (target.equals("this")) name = member.getIdentifier().toString();
            else if (dtos.classes.containsKey(target.substring(target.lastIndexOf('.') + 1))) {
                owner = dtos.classes.get(target.substring(target.lastIndexOf('.') + 1));
                name = member.getIdentifier().toString();
            }
        }
        if (name == null) return null;
        for (Tree member : owner.getMembers()) {
            if (member instanceof MethodTree candidate && candidate.getName().contentEquals(name) && candidate.getBody() != null
                    && candidate.getParameters().size() == call.getArguments().size()
                    && (owner == type || candidate.getModifiers().getFlags().contains(javax.lang.model.element.Modifier.STATIC))) {
                return new Object[] {owner, owner == type ? unit : dtos.units.get(owner), candidate};
            }
        }
        return null;
    }

    /**
     * The type a local holding a read is parsed to: every use of it parses it to that one type,
     * apart from checks of whether it was sent (raw == null, raw.isEmpty()). Null, so text, when
     * any use does anything else.
     */
    // Jackson's JsonNode accessors and what each gives a value as.
    private static final Map<String, String> NODE_KINDS = Map.ofEntries(
            Map.entry("asInt", "int"), Map.entry("intValue", "int"), Map.entry("asLong", "int"), Map.entry("longValue", "int"),
            Map.entry("asDouble", "float"), Map.entry("doubleValue", "float"), Map.entry("asBoolean", "bool"),
            Map.entry("booleanValue", "bool"), Map.entry("asText", "text"), Map.entry("textValue", "text"));
    private static final Set<String> NODE_KEYS = Set.of("get", "path", "has", "hasNonNull", "findValue");

    /**
     * The raw body parsed with Jackson, when what it gives can be followed: readValue(body,
     * Dto.class) gives the class's fields; readTree(body) into a local whose every use reads a
     * constant key (node.get("sku").asText(), node.path("qty").asInt()) gives those keys. Whether
     * it was; anything else stays a raw body read.
     */
    private static boolean parsedBody(CompilationUnitTree unit, MethodTree method, TreePath streamPath, MethodInvocationTree stream,
                                      Map<String, String> inputs, Gaps gaps, Dtos dtos) {
        TreePath parsePath = streamPath.getParentPath();
        if (parsePath == null || !(parsePath.getLeaf() instanceof MethodInvocationTree parse) || parse.getArguments().isEmpty()
                || parse.getArguments().get(0) != stream || !(parse.getMethodSelect() instanceof MemberSelectTree parser)) return false;
        String how = parser.getIdentifier().toString();
        if (how.equals("readValue") && parse.getArguments().size() == 2
                && parse.getArguments().get(1) instanceof MemberSelectTree literal && literal.getIdentifier().contentEquals("class")) {
            Map<String, String> fields = dtos.fields(literal.getExpression(), 0);
            if (fields == null) {
                String[] why = dtos.why(literal.getExpression());
                gaps.add(unit, why[0], why[1], parse);
                return true;
            }
            fields.forEach((name, kind) -> merge(inputs, name, kind));
            gaps.nested.putAll(dtos.nested(literal.getExpression()));
            return true;
        }
        if (!how.equals("readTree") || parsePath.getParentPath() == null
                || !(parsePath.getParentPath().getLeaf() instanceof VariableTree local) || local.getInitializer() != parse) return false;
        Map<String, String> keys = nodeKeys(unit, method, local.getName().toString());
        if (keys == null) return false;
        keys.forEach((name, kind) -> merge(inputs, name, kind));
        return true;
    }

    /** The constant keys a JsonNode local is read with, and their types, or null for any other use. */
    private static Map<String, String> nodeKeys(CompilationUnitTree unit, MethodTree method, String name) {
        Map<String, String> keys = new LinkedHashMap<>();
        boolean[] other = {false};
        new TreePathScanner<Void, Void>() {
            @Override
            public Void visitIdentifier(IdentifierTree node, Void unused) {
                if (!node.getName().contentEquals(name)) return null;
                TreePath path = getCurrentPath();
                TreePath up = path.getParentPath();
                TreePath call = up == null ? null : up.getParentPath();
                if (!(up != null && up.getLeaf() instanceof MemberSelectTree select && select.getExpression() == node
                        && NODE_KEYS.contains(select.getIdentifier().toString())
                        && call != null && call.getLeaf() instanceof MethodInvocationTree read && read.getArguments().size() == 1
                        && read.getArguments().get(0) instanceof LiteralTree literal && literal.getValue() instanceof String key)) {
                    other[0] = true;
                    return null;
                }
                // node.get("qty").asInt(): the accessor called on the value gives its type.
                TreePath outer = call.getParentPath();
                String kind = outer != null && outer.getLeaf() instanceof MemberSelectTree accessor
                        && outer.getParentPath() != null && outer.getParentPath().getLeaf() instanceof MethodInvocationTree
                        ? NODE_KINDS.getOrDefault(accessor.getIdentifier().toString(), "text") : "text";
                merge(keys, key, kind);
                return null;
            }
        }.scan(TreePath.getPath(unit, method.getBody()), null);
        return other[0] ? null : keys;
    }

    static String castLater(CompilationUnitTree unit, MethodTree method, String name) {
        if (method.getBody() == null) return null;
        Set<String> kinds = new java.util.HashSet<>();
        boolean[] other = {false};
        new TreePathScanner<Void, Void>() {
            @Override
            public Void visitIdentifier(IdentifierTree node, Void unused) {
                if (!node.getName().contentEquals(name)) return null;
                Tree parent = getCurrentPath().getParentPath().getLeaf();
                if (parent instanceof AssignmentTree assignment && assignment.getVariable() == node) return null;
                if (parent instanceof MethodInvocationTree call && !call.getArguments().isEmpty() && call.getArguments().get(0) == node
                        && PARSERS.containsKey(call.getMethodSelect().toString())) {
                    kinds.add(PARSERS.get(call.getMethodSelect().toString()));
                } else if (parent instanceof BinaryTree binary
                        && (binary.getKind() == Tree.Kind.EQUAL_TO || binary.getKind() == Tree.Kind.NOT_EQUAL_TO)
                        && (binary.getLeftOperand().getKind() == Tree.Kind.NULL_LITERAL || binary.getRightOperand().getKind() == Tree.Kind.NULL_LITERAL)) {
                    return null;
                } else if (parent instanceof MemberSelectTree select && select.getExpression() == node
                        && Set.of("isEmpty", "isBlank").contains(select.getIdentifier().toString())
                        && getCurrentPath().getParentPath().getParentPath().getLeaf() instanceof MethodInvocationTree check
                        && check.getArguments().isEmpty()) {
                    return null;
                } else {
                    other[0] = true;
                }
                return null;
            }
        }.scan(TreePath.getPath(unit, method.getBody()), null);
        return !other[0] && kinds.size() == 1 ? kinds.iterator().next() : null;
    }

    /**
     * What a handler reads from its HttpServletRequest, as request.getParameter("q"), following the
     * request into the class's own methods, as in return render(request).
     */
    private static void reads(CompilationUnitTree unit, ClassTree type, MethodTree method, List<String> requests, Map<String, String> inputs,
                              Gaps gaps, Dtos dtos, Set<MethodTree> active) {
        Map<String, String> constants = constants(type);
        active.add(method);
        new TreePathScanner<Void, Void>() {
            @Override
            public Void visitIdentifier(IdentifierTree node, Void unused) {
                if (!requests.contains(node.getName().toString())) return null;
                Tree parent = getCurrentPath().getParentPath().getLeaf();
                if (parent instanceof MethodInvocationTree call && call.getArguments().contains(node)) {
                    Object[] found = method(type, unit, call, dtos);
                    MethodTree target = found == null ? null : (MethodTree) found[2];
                    if (target != null && found[1] != null && !active.contains(target) && active.size() < 4) {
                        String name = target.getParameters().get(call.getArguments().indexOf(node)).getName().toString();
                        reads((CompilationUnitTree) found[1], (ClassTree) found[0], target, List.of(name), inputs, gaps, dtos, active);
                        active.remove(target);
                        return null;
                    }
                    gaps.add(unit, "helper", "the request is passed to " + call.getMethodSelect() + ", which isn't followed", call);
                    return null;
                }
                if (!(parent instanceof MemberSelectTree select) || select.getExpression() != node) {
                    gaps.add(unit, "helper", "the request is stored or returned where it may be read", node);
                    return null;
                }
                String called = select.getIdentifier().toString();
                TreePath callPath = getCurrentPath().getParentPath().getParentPath();
                Tree call = callPath == null ? null : callPath.getLeaf();
                if (called.equals("getParameter") || called.equals("getParameterValues")) {
                    MethodInvocationTree invocation = call instanceof MethodInvocationTree item ? item : null;
                    ExpressionTree key = invocation == null || invocation.getArguments().isEmpty() ? null : invocation.getArguments().get(0);
                    String text = key instanceof LiteralTree literal && literal.getValue() instanceof String value ? value
                            : key instanceof IdentifierTree identifier ? constants.get(identifier.getName().toString()) : null;
                    if (text != null) {
                        Tree outer = callPath.getParentPath() == null ? null : callPath.getParentPath().getLeaf();
                        String kind = outer instanceof MethodInvocationTree parse && !parse.getArguments().isEmpty()
                                && parse.getArguments().get(0) == invocation ? PARSERS.get(parse.getMethodSelect().toString()) : null;
                        // String raw = request.getParameter("n"), parsed later: Integer.parseInt(raw).
                        if (kind == null && outer instanceof VariableTree local && local.getInitializer() == invocation) {
                            kind = castLater(unit, method, local.getName().toString());
                        }
                        merge(inputs, text, kind == null ? "text" : kind);
                    } else {
                        gaps.add(unit, "dynamic_key", "request." + called + " is called with a key that isn't a constant", call);
                    }
                } else if (("getInputStream".equals(called) || "getReader".equals(called)) && call instanceof MethodInvocationTree stream
                        && parsedBody(unit, method, callPath, stream, inputs, gaps, dtos)) {
                    return null;  // Parsed with Jackson, and every read of what it parsed followed.
                } else if (RAW_READS.contains(called)) {
                    String code = Set.of("getParameterMap", "getParameterNames").contains(called) ? "container_whole"
                            : called.equals("getAttribute") ? "helper" : "raw_body";
                    gaps.add(unit, code, "request." + called + "() reads " + (code.equals("helper")
                            ? "attributes that may hold input" : code.equals("raw_body") ? "the raw body or URL" : "every parameter"), call);
                }
                return null;
            }
        }.scan(TreePath.getPath(unit, method.getBody()), null);
    }
}
