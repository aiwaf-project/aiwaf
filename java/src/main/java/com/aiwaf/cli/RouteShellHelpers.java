package com.aiwaf.cli;

import org.springframework.core.annotation.AnnotatedElementUtils;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public final class RouteShellHelpers {
    private RouteShellHelpers() {}

    public static List<String> collectRoutes(Object... handlers) {
        List<String> routes = new ArrayList<>();
        if (handlers == null) return routes;
        for (Object handler : handlers) {
            if (handler == null) continue;
            Class<?> handlerType = handler.getClass();
            String[] classPaths = extractPaths(AnnotatedElementUtils.findMergedAnnotation(handlerType, RequestMapping.class));
            if (classPaths.length == 0) classPaths = new String[]{""};
            for (Method method : handlerType.getMethods()) {
                Mapping mapping = extractMethodMapping(method);
                if (mapping == null) continue;
                String[] methodPaths = mapping.paths().length == 0 ? new String[]{""} : mapping.paths();
                for (String classPath : classPaths) {
                    for (String methodPath : methodPaths) {
                        routes.add(joinPaths(classPath, methodPath));
                    }
                }
            }
        }
        return routes;
    }

    public static RouteNode buildTree(Collection<String> routes) {
        RouteNode root = new RouteNode("/", null);
        if (routes == null) return root;
        for (String route : routes) {
            String normalized = normalizePath(route);
            String[] segments = normalized.substring(1).split("/");
            RouteNode cursor = root;
            if (normalized.equals("/")) {
                root.setEndpoint(true);
                continue;
            }
            for (String segment : segments) {
                if (segment.isBlank()) continue;
                RouteNode next = cursor.child(segment);
                if (next == null) {
                    next = new RouteNode(segment, cursor);
                    cursor.children().put(segment, next);
                }
                cursor = next;
            }
            cursor.setEndpoint(true);
        }
        return root;
    }

    public static RouteNode resolveTarget(RouteNode current, String token) {
        if (current == null) return null;
        if (token == null || token.isBlank() || ".".equals(token)) return current;
        if ("..".equals(token)) return current.parent() == null ? current : current.parent();
        try {
            int idx = Integer.parseInt(token.trim());
            RouteNode byIndex = current.childAt(idx);
            if (byIndex != null) return byIndex;
        } catch (NumberFormatException ignored) {
        }
        RouteNode byName = current.child(token.trim());
        return byName == null ? current : byName;
    }

    public static String routePath(RouteNode node) {
        if (node == null) return "/";
        List<String> parts = new ArrayList<>();
        RouteNode cursor = node;
        while (cursor != null && cursor.parent() != null) {
            parts.add(0, cursor.segment());
            cursor = cursor.parent();
        }
        if (parts.isEmpty()) return "/";
        return "/" + String.join("/", parts) + "/";
    }

    public static boolean exemptCurrentPath(AiwafManager manager, RouteNode current, String reason) {
        if (manager == null || current == null) return false;
        return manager.addPathExemption(routePath(current), reason);
    }

    private static Mapping extractMethodMapping(Method method) {
        GetMapping get = AnnotatedElementUtils.findMergedAnnotation(method, GetMapping.class);
        if (get != null) return new Mapping(pathsFrom(get.value(), get.path()), new RequestMethod[]{RequestMethod.GET});
        PostMapping post = AnnotatedElementUtils.findMergedAnnotation(method, PostMapping.class);
        if (post != null) return new Mapping(pathsFrom(post.value(), post.path()), new RequestMethod[]{RequestMethod.POST});
        PutMapping put = AnnotatedElementUtils.findMergedAnnotation(method, PutMapping.class);
        if (put != null) return new Mapping(pathsFrom(put.value(), put.path()), new RequestMethod[]{RequestMethod.PUT});
        PatchMapping patch = AnnotatedElementUtils.findMergedAnnotation(method, PatchMapping.class);
        if (patch != null) return new Mapping(pathsFrom(patch.value(), patch.path()), new RequestMethod[]{RequestMethod.PATCH});
        DeleteMapping delete = AnnotatedElementUtils.findMergedAnnotation(method, DeleteMapping.class);
        if (delete != null) return new Mapping(pathsFrom(delete.value(), delete.path()), new RequestMethod[]{RequestMethod.DELETE});
        RequestMapping req = AnnotatedElementUtils.findMergedAnnotation(method, RequestMapping.class);
        if (req != null) return new Mapping(extractPaths(req), req.method());
        return null;
    }

    private static String[] extractPaths(RequestMapping mapping) {
        if (mapping == null) return new String[0];
        return pathsFrom(mapping.value(), mapping.path());
    }

    private static String[] pathsFrom(String[] value, String[] path) {
        if (path != null && path.length > 0) return path;
        if (value != null && value.length > 0) return value;
        return new String[0];
    }

    private static String joinPaths(String left, String right) {
        String a = left == null ? "" : left.trim();
        String b = right == null ? "" : right.trim();
        if (a.isEmpty() && b.isEmpty()) return "/";
        if (a.isEmpty()) return normalizePath(b);
        if (b.isEmpty()) return normalizePath(a);
        return normalizePath(a + "/" + b);
    }

    private static String normalizePath(String path) {
        if (path == null || path.isBlank()) return "/";
        String normalized = path.replaceAll("/+", "/");
        if (!normalized.startsWith("/")) normalized = "/" + normalized;
        if (normalized.length() > 1 && normalized.endsWith("/")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        return normalized;
    }

    private record Mapping(String[] paths, RequestMethod[] methods) {}
}
