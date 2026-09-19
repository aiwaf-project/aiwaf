package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class HttpCoreParityTest {

    @Test
    void normalize_headers_lowercases_and_hyphenates() {
        Map<String, String> out = HttpCore.normalizeHeaders(
                Map.of("User_Agent", "UA", "ACCEPT", "text/html", "X", "")
        );
        assertEquals("UA", out.get("user-agent"));
        assertEquals("text/html", out.get("accept"));
        assertEquals("", out.get("x"));
    }

    @Test
    void normalize_wsgi_environ_extracts_http_and_protocol() {
        Map<String, String> out = HttpCore.normalizeWsgiEnviron(Map.of(
                "HTTP_USER_AGENT", "UA",
                "HTTP_ACCEPT_LANGUAGE", "en-US",
                "CONTENT_TYPE", "application/json",
                "SERVER_PROTOCOL", "HTTP/2",
                "SOME_OTHER", "x"
        ));
        assertEquals("UA", out.get("user-agent"));
        assertEquals("en-US", out.get("accept-language"));
        assertEquals("application/json", out.get("content-type"));
        assertEquals("HTTP/2", out.get("server-protocol"));
        assertFalse(out.containsKey("some-other"));
    }
}
