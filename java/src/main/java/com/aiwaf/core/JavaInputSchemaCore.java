package com.aiwaf.core;

import com.sun.source.tree.AnnotationTree;
import com.sun.source.tree.AssignmentTree;
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
 * <p>Anything not understood makes the list incomplete rather than guessed: the request passed to
 * another method, the parameter map or raw body read, a {@code Map} or unknown body type, a
 * class with {@code @JsonAnySetter}, or {@code @InitBinder} and {@code @ModelAttribute} methods.
 */
public final class JavaInputSchemaCore {
    /** One handler's inputs, labelled Class.method as the audit labels its findings. */
    public record Inputs(String function, String file, Map<String, String> inputs, boolean complete) {}

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
        final Set<String> enums = new java.util.HashSet<>();

        Dtos(List<CompilationUnitTree> units) {
            for (CompilationUnitTree unit : units) {
                new TreeScanner<Void, Void>() {
                    @Override
                    public Void visitClass(ClassTree node, Void unused) {
                        classes.putIfAbsent(node.getSimpleName().toString(), node);
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

        boolean isEnum(Tree type) {
            return enums.contains(simpleName(type));
        }
    }

    /** Each handler's inputs in the parsed units, with each unit's display name. */
    public static List<Inputs> inputs(List<CompilationUnitTree> units, java.util.function.Function<CompilationUnitTree, String> display) {
        Dtos dtos = new Dtos(units);
        List<Inputs> results = new ArrayList<>();
        for (CompilationUnitTree unit : units) {
            String file = display.apply(unit);
            new TreeScanner<Void, Void>() {
                @Override
                public Void visitClass(ClassTree type, Void unused) {
                    handlers(unit, type, file, dtos, results);
                    return super.visitClass(type, unused);
                }
            }.scan(unit, null);
        }
        return results;
    }

    private static void handlers(CompilationUnitTree unit, ClassTree type, String file, Dtos dtos, List<Inputs> results) {
        // Methods every handler of the class runs first, which may read anything.
        boolean shared = type.getMembers().stream().anyMatch(member -> member instanceof MethodTree method
                && (annotations(method.getModifiers().getAnnotations()).containsKey("InitBinder")
                    || annotations(method.getModifiers().getAnnotations()).containsKey("ModelAttribute")));
        for (Tree member : type.getMembers()) {
            if (!(member instanceof MethodTree method) || method.getBody() == null) continue;
            Map<String, AnnotationTree> marks = annotations(method.getModifiers().getAnnotations());
            boolean mapped = MAPPINGS.stream().anyMatch(marks::containsKey);
            boolean servlet = SERVLET_METHODS.contains(method.getName().toString())
                    && method.getParameters().stream().anyMatch(parameter -> REQUEST_TYPES.contains(simpleName(parameter.getType())));
            if (!mapped && !servlet) continue;
            Map<String, String> inputs = new LinkedHashMap<>();
            boolean[] complete = {!shared};
            List<String> requests = new ArrayList<>();
            for (VariableTree parameter : method.getParameters()) {
                parameter(parameter, dtos, inputs, complete, requests);
            }
            if (!requests.isEmpty()) reads(unit, type, method, requests, inputs, complete, new java.util.HashSet<>());
            results.add(new Inputs(type.getSimpleName() + "." + method.getName(), file, inputs, complete[0]));
        }
    }

    private static void parameter(VariableTree parameter, Dtos dtos, Map<String, String> inputs, boolean[] complete, List<String> requests) {
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
        if (Set.of("WebRequest", "NativeWebRequest", "MultipartFile", "MultipartHttpServletRequest", "HttpEntity", "RequestEntity",
                "InputStream", "Reader").contains(type) || marks.containsKey("RequestPart")) {
            complete[0] = false;
            return;
        }
        String kind = scalar(parameter.getType());
        if (marks.containsKey("RequestParam") || (kind != null && !marks.containsKey("RequestBody") && !marks.containsKey("ModelAttribute"))) {
            if (kind == null && !dtos.isEnum(parameter.getType())) {
                complete[0] = false;  // A map of every parameter, or a type that isn't known.
                return;
            }
            String named = marks.containsKey("RequestParam") ? annotationName(marks.get("RequestParam")) : null;
            merge(inputs, named != null ? named : name, kind == null ? "text" : kind);
            return;
        }
        // A @RequestBody, or a @ModelAttribute or other object bound from the form and query.
        Map<String, String> fields = dtos.fields(parameter.getType(), 0);
        if (fields == null) {
            complete[0] = false;
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

    /** A method of the class a call names, as helper(request) or this.helper(request), with as many parameters. */
    private static MethodTree method(ClassTree type, MethodInvocationTree call) {
        ExpressionTree select = call.getMethodSelect();
        String name = select instanceof IdentifierTree identifier ? identifier.getName().toString()
                : select instanceof MemberSelectTree member && member.getExpression().toString().equals("this") ? member.getIdentifier().toString() : null;
        if (name == null) return null;
        for (Tree member : type.getMembers()) {
            if (member instanceof MethodTree candidate && candidate.getName().contentEquals(name) && candidate.getBody() != null
                    && candidate.getParameters().size() == call.getArguments().size()) {
                return candidate;
            }
        }
        return null;
    }

    /**
     * What a handler reads from its HttpServletRequest, as request.getParameter("q"), following the
     * request into the class's own methods, as in return render(request).
     */
    private static void reads(CompilationUnitTree unit, ClassTree type, MethodTree method, List<String> requests, Map<String, String> inputs,
                              boolean[] complete, Set<MethodTree> active) {
        Map<String, String> constants = constants(type);
        active.add(method);
        new TreePathScanner<Void, Void>() {
            @Override
            public Void visitIdentifier(IdentifierTree node, Void unused) {
                if (!requests.contains(node.getName().toString())) return null;
                Tree parent = getCurrentPath().getParentPath().getLeaf();
                if (parent instanceof MethodInvocationTree call && call.getArguments().contains(node)) {
                    MethodTree target = method(type, call);
                    if (target != null && !active.contains(target) && active.size() < 4) {
                        String name = target.getParameters().get(call.getArguments().indexOf(node)).getName().toString();
                        reads(unit, type, target, List.of(name), inputs, complete, active);
                        active.remove(target);
                        return null;
                    }
                }
                if (!(parent instanceof MemberSelectTree select) || select.getExpression() != node) {
                    complete[0] = false;  // The request goes somewhere that may read it.
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
                        merge(inputs, text, kind == null ? "text" : kind);
                    } else {
                        complete[0] = false;
                    }
                } else if (RAW_READS.contains(called)) {
                    complete[0] = false;
                }
                return null;
            }
        }.scan(TreePath.getPath(unit, method.getBody()), null);
    }
}
