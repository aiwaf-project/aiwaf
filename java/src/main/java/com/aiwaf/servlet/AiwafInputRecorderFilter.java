package com.aiwaf.servlet;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.FilterConfig;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletRequestWrapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Records which query and form parameters each handler reads, while tests or staging traffic run.
 * Set {@code AIWAF_RECORD_INPUTS} (environment or system property) to a file path, or to 1 for
 * {@code .aiwaf/observed_inputs.jsonl}, and register this filter; without it the filter only passes
 * the request on.
 *
 * <p>Each request appends a line per parameter read with {@code getParameter} or
 * {@code getParameterValues}: {@code {"route", "method", "key"}}, the route being Spring's matched
 * pattern when there is one. Reads of the whole parameter map and of the body aren't recorded,
 * since Spring's binders and message converters make them for every handler; the source scan
 * reports handlers that make them themselves. {@code init} reads the file (see InputSourcesCore).
 */
public final class AiwafInputRecorderFilter implements Filter {
    public static final String ENVIRONMENT = "AIWAF_RECORD_INPUTS";
    public static final String DEFAULT_PATH = ".aiwaf/observed_inputs.jsonl";
    private static final String BEST_MATCHING_PATTERN = "org.springframework.web.servlet.HandlerMapping.bestMatchingPattern";
    // Read by CSRF protection on every form post, not by the handler.
    private static final Set<String> IGNORED = Set.of("_csrf", "csrf_token", "csrfmiddlewaretoken");
    private static final ObjectMapper JSON = new ObjectMapper();

    private final Path target;

    public AiwafInputRecorderFilter() {
        this(destination(System.getProperty(ENVIRONMENT, System.getenv(ENVIRONMENT))));
    }

    public AiwafInputRecorderFilter(Path target) {
        this.target = target;
    }

    /** Where to record for a setting's value, or null when recording is off. */
    public static Path destination(String value) {
        String raw = value == null ? "" : value.trim();
        String lowered = raw.toLowerCase(Locale.ROOT);
        if (raw.isEmpty() || Set.of("0", "false", "no", "off").contains(lowered)) return null;
        if (Set.of("1", "true", "yes", "on").contains(lowered)) return Path.of(DEFAULT_PATH);
        return Path.of(raw);
    }

    @Override
    public void init(FilterConfig filterConfig) {
        // Configured by AIWAF_RECORD_INPUTS, or the constructor.
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        if (target == null || !(request instanceof HttpServletRequest http)) {
            chain.doFilter(request, response);
            return;
        }
        Set<String> keys = new TreeSet<>();
        HttpServletRequestWrapper recording = new HttpServletRequestWrapper(http) {
            @Override
            public String getParameter(String name) {
                note(name);
                return super.getParameter(name);
            }

            @Override
            public String[] getParameterValues(String name) {
                note(name);
                return super.getParameterValues(name);
            }

            private void note(String name) {
                if (name != null && !IGNORED.contains(name)) keys.add(name);
            }
        };
        try {
            chain.doFilter(recording, response);
        } finally {
            Object pattern = http.getAttribute(BEST_MATCHING_PATTERN);
            String route = pattern != null ? pattern.toString()
                    : http.getServletPath() + (http.getPathInfo() == null ? "" : http.getPathInfo());
            save(route, http.getMethod(), keys);
        }
    }

    private void save(String route, String method, Set<String> keys) {
        if (keys.isEmpty() || route == null || route.isEmpty()) return;
        StringBuilder lines = new StringBuilder();
        for (String key : keys) {
            Map<String, String> line = new LinkedHashMap<>();
            line.put("route", route);
            line.put("method", method == null ? "GET" : method.toUpperCase(Locale.ROOT));
            line.put("key", key);
            try {
                lines.append(JSON.writeValueAsString(line)).append('\n');
            } catch (IOException ignored) {
                // A map of strings always serializes.
            }
        }
        try {
            if (target.getParent() != null) Files.createDirectories(target.getParent());
            Files.writeString(target, lines.toString(), StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException ex) {
            System.err.println("AIWAF couldn't record inputs to " + target + ": " + ex.getMessage());
        }
    }
}
