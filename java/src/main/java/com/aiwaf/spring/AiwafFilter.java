package com.aiwaf.spring;

import com.aiwaf.core.AiwafLoggingCore;
import com.aiwaf.core.AiwafRequest;
import com.aiwaf.core.AiwafDecision;
import com.aiwaf.core.AiwafEngine;
import com.aiwaf.core.LegitimateRouteKeywordsCore;
import com.aiwaf.core.ServletRequestMapper;
import com.aiwaf.core.BufferedServletRequest;
import com.aiwaf.core.PathManifestCore;
import com.aiwaf.spring.support.AiwafRouteDecisions;
import jakarta.servlet.AsyncEvent;
import jakarta.servlet.AsyncListener;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.core.annotation.AnnotatedElementUtils;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.method.HandlerMethod;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;

public final class AiwafFilter extends OncePerRequestFilter {
    private static final Set<String> SUPPORTED_MIDDLEWARES = Set.of(
            "ip_keyword_block", "rate_limit", "honeypot",
            "header_validation", "geo_block", "uuid_tamper", "ai_anomaly"
    );

    private final AiwafEngine engine;
    private final List<RoutePolicy> routePolicies;

    public AiwafFilter(AiwafEngine engine) {
        this(engine, new Object[0]);
    }

    public AiwafFilter(AiwafEngine engine, Object... handlers) {
        this.engine = engine;
        this.routePolicies = buildPolicies(handlers);
        enrichLegitimateKeywords(engine, handlers);
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        if (Boolean.TRUE.equals(request.getAttribute("aiwaf.already.checked"))) {
            filterChain.doFilter(request, response);
            return;
        }
        long start = System.currentTimeMillis();
        Set<String> disabledMiddlewares = resolveDisabledMiddlewares(request);
        BufferedServletRequest.Result buffered = BufferedServletRequest.prepare(request, engine.config());
        if (buffered.tooLarge()) {
            response.sendError(413, "Request body too large");
            return;
        }
        HttpServletRequest inspectedRequest = buffered.request();
        AiwafRequest mappedRequest = ServletRequestMapper.from(
                inspectedRequest, disabledMiddlewares, engine.config(), buffered.preview());
        PathVariableCandidate routeUuid = resolveUuidPathVariable(inspectedRequest);
        final AiwafRequest aiwafReq = routeUuid == null
                ? mappedRequest
                : mappedRequest.withQueryParameter(routeUuid.name(), routeUuid.value());
        AiwafDecision decision = engine.evaluateBeforeResponse(aiwafReq);
        if (!decision.allowed()) {
            response.sendError(decision.statusCode(), decision.reason());
            AiwafLoggingCore.log(engine.config(), aiwafReq, decision, decision.statusCode(), System.currentTimeMillis() - start, 0);
            return;
        }
        filterChain.doFilter(inspectedRequest, response);
        if (inspectedRequest.isAsyncStarted()) {
            AtomicBoolean recorded = new AtomicBoolean();
            inspectedRequest.getAsyncContext().addListener(new AsyncListener() {
                private void record(AsyncEvent event) {
                    if (!recorded.compareAndSet(false, true)) return;
                    HttpServletResponse completed = (HttpServletResponse) event.getSuppliedResponse();
                    int status = completed == null ? response.getStatus() : completed.getStatus();
                    engine.recordCommittedResponse(aiwafReq, status);
                    AiwafLoggingCore.log(engine.config(), aiwafReq, decision, status,
                            System.currentTimeMillis() - start, 0);
                }

                @Override public void onComplete(AsyncEvent event) { record(event); }
                @Override public void onTimeout(AsyncEvent event) { record(event); }
                @Override public void onError(AsyncEvent event) { record(event); }
                @Override public void onStartAsync(AsyncEvent event) {
                    event.getAsyncContext().addListener(this);
                }
            });
            return;
        }
        int handlerStatus = response.getStatus();
        long responseTimeMs = System.currentTimeMillis() - start;
        if (response.isCommitted()) {
            engine.recordCommittedResponse(aiwafReq, handlerStatus);
            AiwafLoggingCore.log(engine.config(), aiwafReq, decision, handlerStatus, responseTimeMs, 0);
            return;
        }
        AiwafDecision finalDecision = engine.evaluateAfterResponse(aiwafReq, handlerStatus, responseTimeMs);
        if (!finalDecision.allowed()) {
            response.reset();
            response.sendError(finalDecision.statusCode(), finalDecision.reason());
        }
        AiwafLoggingCore.log(engine.config(), aiwafReq, finalDecision, response.getStatus(), responseTimeMs, 0);
    }

    private Set<String> resolveDisabledMiddlewares(HttpServletRequest request) {
        Object fromInterceptor = request.getAttribute("aiwaf.disabled.middlewares");
        if (fromInterceptor instanceof Set<?> s) {
            Set<String> disabled = new HashSet<>();
            for (Object item : s) {
                if (item != null) {
                    disabled.add(String.valueOf(item));
                }
            }
            return disabled;
        }
        String method = normalizeMethod(request.getMethod());
        String path = request.getRequestURI() == null ? "/" : request.getRequestURI();
        for (RoutePolicy policy : routePolicies) {
            if (policy.matches(method, path)) {
                Set<String> disabled = new HashSet<>();
                for (String middleware : SUPPORTED_MIDDLEWARES) {
                    if (!AiwafRouteDecisions.shouldApply(path, policy.handler(), middleware, engine.config().pathRules)) {
                        disabled.add(middleware);
                    }
                }
                return disabled;
            }
        }
        return Set.of();
    }

    private static List<RoutePolicy> buildPolicies(Object... handlers) {
        List<RoutePolicy> out = new ArrayList<>();
        if (handlers == null || handlers.length == 0) {
            return out;
        }
        for (Object handler : handlers) {
            if (handler == null) continue;
            for (PathManifestCore.RouteInfo route : PathManifestCore.discoverControllerRoutes(handler)) {
                HandlerMethod hm = new HandlerMethod(handler, route.method());
                Set<String> methods = new HashSet<>(route.httpMethods());
                String routePath = route.path();
                out.add(new RoutePolicy(routePath, toRegexPath(routePath), methods, hm));
            }
        }
        return out;
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
        if (mapping == null) {
            return new String[0];
        }
        return pathsFrom(mapping.value(), mapping.path());
    }

    private static String[] pathsFrom(String[] value, String[] path) {
        if (path != null && path.length > 0) {
            return path;
        }
        if (value != null && value.length > 0) {
            return value;
        }
        return new String[0];
    }

    private static String normalizeMethod(String method) {
        return method == null ? "" : method.toUpperCase(Locale.ROOT);
    }

    private static String normalizePath(String path) {
        String normalized = path.replaceAll("/+", "/");
        if (!normalized.startsWith("/")) {
            normalized = "/" + normalized;
        }
        return normalized.length() > 1 && normalized.endsWith("/")
                ? normalized.substring(0, normalized.length() - 1)
                : normalized;
    }

    private static Pattern toRegexPath(String path) {
        StringBuilder regex = new StringBuilder("^");
        String normalized = normalizePath(path);
        for (int i = 0; i < normalized.length(); i++) {
            char c = normalized.charAt(i);
            if (c == '{') {
                while (i < normalized.length() && normalized.charAt(i) != '}') {
                    i++;
                }
                regex.append("[^/]+");
            } else {
                if ("\\.[]{}()+-*?^$|".indexOf(c) >= 0) {
                    regex.append('\\');
                }
                regex.append(c);
            }
        }
        regex.append("$");
        return Pattern.compile(regex.toString());
    }

    private PathVariableCandidate resolveUuidPathVariable(HttpServletRequest request) {
        String method = normalizeMethod(request.getMethod());
        String path = request.getRequestURI() == null ? "/" : request.getRequestURI();
        for (RoutePolicy policy : routePolicies) {
            if (!policy.matches(method, path)) continue;
            PathVariableCandidate candidate = policy.uuidCandidate(path, engine.config().uuidParameterNames);
            if (candidate != null) return candidate;
        }
        return null;
    }

    private static void enrichLegitimateKeywords(AiwafEngine engine, Object... handlers) {
        if (engine == null || engine.config() == null || handlers == null || handlers.length == 0) {
            return;
        }
        Set<String> out = engine.config().legitimatePathKeywords;
        Set<Class<?>> types = new HashSet<>();
        for (Object handler : handlers) {
            if (handler == null) continue;
            Class<?> type = handler.getClass();
             types.add(type);
            addTokenized(out, type.getSimpleName());

            RequestMapping classMapping = AnnotatedElementUtils.findMergedAnnotation(type, RequestMapping.class);
            for (String classPath : extractPaths(classMapping)) {
                addPathTokens(out, classPath);
            }

            for (Method method : type.getMethods()) {
                addTokenized(out, method.getName());
                Mapping mapping = extractMethodMapping(method);
                if (mapping == null) continue;
                for (String methodPath : mapping.paths()) {
                    addPathTokens(out, methodPath);
                }
            }
        }
        LegitimateRouteKeywordsCore.mergeInto(out, LegitimateRouteKeywordsCore.fromHandlerClasses(types));
    }

    private static void addPathTokens(Set<String> out, String rawPath) {
        if (rawPath == null || rawPath.isBlank()) return;
        String normalized = rawPath.toLowerCase(Locale.ROOT);
        StringBuilder literal = new StringBuilder();
        boolean inVar = false;
        for (int i = 0; i < normalized.length(); i++) {
            char c = normalized.charAt(i);
            if (c == '{') {
                inVar = true;
            } else if (c == '}') {
                inVar = false;
            } else if (!inVar) {
                literal.append(c);
            }
        }
        addTokenized(out, literal.toString());
    }

    private static void addTokenized(Set<String> out, String text) {
        if (text == null || text.isBlank()) return;
        String[] parts = text.toLowerCase(Locale.ROOT).split("[^a-z0-9]+");
        for (String part : parts) {
            if (part.length() >= 3) {
                out.add(part);
                if (!part.endsWith("s")) out.add(part + "s");
            }
        }
    }

    private record Mapping(String[] paths, RequestMethod[] methods) {}

    private record PathVariableCandidate(String name, String value) {}

    private record RoutePolicy(String template, Pattern pathPattern, Set<String> methods, HandlerMethod handler) {
        boolean matches(String method, String path) {
            if (!methods.isEmpty() && !methods.contains(method)) {
                return false;
            }
            return pathPattern.matcher(path).matches();
        }

        PathVariableCandidate uuidCandidate(String actualPath, Set<String> configuredNames) {
            String[] templateParts = normalizePath(template).substring(1).split("/", -1);
            String[] actualParts = normalizePath(actualPath).substring(1).split("/", -1);
            if (templateParts.length != actualParts.length) return null;
            for (int i = 0; i < templateParts.length; i++) {
                String part = templateParts[i];
                if (!part.startsWith("{") || !part.endsWith("}")) continue;
                String name = part.substring(1, part.length() - 1);
                int constraint = name.indexOf(':');
                if (constraint >= 0) name = name.substring(0, constraint);
                String candidateName = name;
                boolean configured = configuredNames.stream()
                        .filter(java.util.Objects::nonNull)
                        .anyMatch(value -> value.equalsIgnoreCase(candidateName));
                if (configured) return new PathVariableCandidate(name, actualParts[i]);
            }
            return null;
        }
    }
}
