package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HeaderValidationCoreParityTest {

    @Test
    void valid_browser_headers_pass() {
        Map<String, String> headers = new HashMap<>();
        headers.put("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit");
        headers.put("Accept", "text/html,application/xml");
        headers.put("Accept-Language", "en-US,en;q=0.9");
        headers.put("Accept-Encoding", "gzip, deflate, br");
        headers.put("Connection", "keep-alive");

        String issue = HeaderValidationCore.validate(
                headers, "GET", "HTTP/1.1",
                List.of("user-agent", "accept"), 3,
                32 * 1024, 100, 500, 4096
        );
        assertNull(issue);
    }

    @Test
    void missing_required_header_fails() {
        Map<String, String> headers = new HashMap<>();
        headers.put("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");
        String issue = HeaderValidationCore.validate(
                headers, "GET", "HTTP/1.1",
                List.of("user-agent", "accept"), 3,
                32 * 1024, 100, 500, 4096
        );
        assertNotNull(issue);
        assertTrue(issue.contains("Missing required headers"));
    }

    @Test
    void suspicious_user_agent_fails() {
        Map<String, String> headers = new HashMap<>();
        headers.put("User-Agent", "python-requests/2.28");
        headers.put("Accept", "*/*");
        String issue = HeaderValidationCore.validate(
                headers, "GET", "HTTP/1.1",
                List.of("user-agent", "accept"), 3,
                32 * 1024, 100, 500, 4096
        );
        assertNotNull(issue);
        assertTrue(issue.toLowerCase().contains("suspicious user agent"));
    }

    @Test
    void suspicious_combo_fails() {
        Map<String, String> headers = new HashMap<>();
        headers.put("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");
        headers.put("Accept", "*/*");
        String issue = HeaderValidationCore.validate(
                headers, "GET", "HTTP/1.1",
                List.of("user-agent", "accept"), 3,
                32 * 1024, 100, 500, 4096
        );
        assertNotNull(issue);
        assertTrue(issue.toLowerCase().contains("suspicious headers"));
    }

    @Test
    void required_headers_can_be_method_specific() {
        Map<String, String> headers = new HashMap<>();
        headers.put("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");
        headers.put("Accept", "text/html");

        String issue = HeaderValidationCore.validate(
                headers, "GET", "HTTP/1.1",
                List.of("accept"), 1,
                32 * 1024, 100, 500, 4096
        );
        assertNull(issue);
    }

    @Test
    void empty_required_headers_do_not_fall_back_to_default() {
        Map<String, String> headers = new HashMap<>();
        headers.put("User-Agent", "EmailScanner/1.0");

        String issue = HeaderValidationCore.validate(
                headers, "HEAD", "HTTP/1.1",
                List.of(), 3,
                32 * 1024, 100, 500, 4096
        );
        assertNull(issue);
    }

    @Test
    void required_header_names_accept_http_underscore_format() {
        Map<String, String> headers = new HashMap<>();
        headers.put("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");
        headers.put("Accept", "text/html");

        String issue = HeaderValidationCore.validate(
                headers, "GET", "HTTP/1.1",
                List.of("HTTP_USER_AGENT", "HTTP_ACCEPT"), 1,
                32 * 1024, 100, 500, 4096
        );
        assertNull(issue);
    }
}
