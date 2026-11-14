package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;

import java.nio.charset.StandardCharsets;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RequestBoundarySecurityTest {

    @Test
    void ignores_forwarding_and_country_headers_from_untrusted_peer() {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.setRemoteAddr("198.51.100.7");
        request.addHeader("X-Forwarded-For", "10.0.0.1");
        request.addHeader("X-Country", "US");
        AiwafConfig config = new AiwafConfig();

        AiwafRequest mapped = ServletRequestMapper.from(request, Set.of(), config);

        assertEquals("198.51.100.7", mapped.ip());
        assertEquals("", mapped.country());
    }

    @Test
    void malformed_forwarded_hop_does_not_expose_client_supplied_values() {
        for (String malformed : new String[] {"garbage", "unknown", "", "203.0.113.5:4711"}) {
            MockHttpServletRequest request = new MockHttpServletRequest();
            request.setRemoteAddr("127.0.0.1");
            request.addHeader("X-Forwarded-For", "198.51.100.66, " + malformed);

            AiwafRequest mapped = ServletRequestMapper.from(request, Set.of(), new AiwafConfig());

            assertEquals("127.0.0.1", mapped.ip(), "hop: '" + malformed + "'");
        }
    }

    @Test
    void forwarded_walk_returns_rightmost_untrusted_hop_past_trusted_proxies() {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.setRemoteAddr("127.0.0.1");
        request.addHeader("X-Forwarded-For", "198.51.100.66, 203.0.113.5, 127.0.0.2");

        AiwafRequest mapped = ServletRequestMapper.from(request, Set.of(), new AiwafConfig());

        assertEquals("203.0.113.5", mapped.ip());
    }

    @Test
    void buffers_body_for_downstream_and_detects_raw_java_stream() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest();
        byte[] body = {(byte) 0xac, (byte) 0xed, 0, 5, 1, 2, 3};
        request.setContent(body);
        request.setContentType("application/octet-stream");

        BufferedServletRequest.Result result = BufferedServletRequest.prepare(request, new AiwafConfig());

        assertFalse(result.tooLarge());
        assertEquals("aced0005", result.preview());
        assertEquals(body.length, result.request().getInputStream().readAllBytes().length);
    }

    @Test
    void rejects_declared_oversized_body_without_consuming_it() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.setContent("payload-is-too-large".getBytes(StandardCharsets.UTF_8));
        AiwafConfig config = new AiwafConfig();
        config.maxRequestBodyBytes = 10;

        assertTrue(BufferedServletRequest.prepare(request, config).tooLarge());
    }
}
