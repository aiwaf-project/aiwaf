package com.aiwaf.core;

import jakarta.servlet.ReadListener;
import jakarta.servlet.ServletInputStream;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletRequestWrapper;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

public final class BufferedServletRequest extends HttpServletRequestWrapper {
    private final byte[] body;

    private BufferedServletRequest(HttpServletRequest request, byte[] body) {
        super(request);
        this.body = body;
    }

    public static Result prepare(HttpServletRequest request, AiwafConfig config) throws IOException {
        int max = Math.max(1, config.maxRequestBodyBytes);
        long declared = request.getContentLengthLong();
        if (declared > max) return new Result(request, "", true);

        String contentType = request.getContentType() == null ? "" : request.getContentType().toLowerCase(Locale.ROOT);
        if (config.requestBodyInspectionEnabled && contentType.startsWith("multipart/form-data") && declared >= 0) {
            // The container parses the form, and the payload check reads its parts.
            return new Result(request, MultipartPreview.of(request, config.requestBodyInspectionBytes), false);
        }
        if (!config.requestBodyInspectionEnabled || contentType.startsWith("multipart/")
                || contentType.startsWith("application/x-www-form-urlencoded")) {
            if (declared < 0 && (contentType.startsWith("multipart/")
                    || contentType.startsWith("application/x-www-form-urlencoded"))) {
                return new Result(request, "", true);
            }
            return new Result(request, "", false);
        }

        byte[] bytes = request.getInputStream().readNBytes(max + 1);
        if (bytes.length > max) return new Result(request, "", true);
        if ("block".equals(config.sqlInjectionMode) && bytes.length > config.requestBodyInspectionBytes) {
            return new Result(request, "", true);
        }
        BufferedServletRequest wrapped = new BufferedServletRequest(request, bytes);
        // A compressed body, when allowCompressedRequestBodies lets one in, is checked inflated;
        // the application still gets it as sent. One that can't be read can't be checked.
        byte[] inspected = bytes;
        String encoding = request.getHeader("Content-Encoding") == null ? "" : request.getHeader("Content-Encoding").trim().toLowerCase(Locale.ROOT);
        if (!encoding.isEmpty() && !encoding.equals("identity")) {
            byte[] inflated = encoding.equals("gzip") || encoding.equals("x-gzip") || encoding.equals("deflate")
                    ? RceCore.inflate(bytes, config.requestBodyInspectionBytes + 1) : null;
            if (inflated == null || inflated.length > config.requestBodyInspectionBytes) return new Result(request, "", true);
            inspected = inflated;
        }
        int previewLength = Math.min(inspected.length, Math.max(0, config.requestBodyInspectionBytes));
        boolean javaObject = inspected.length >= 4 && (inspected[0] & 0xff) == 0xac && (inspected[1] & 0xff) == 0xed
                && inspected[2] == 0 && inspected[3] == 5;
        // Other binary objects, as in pickle, are kept as base64, which the check decodes, since
        // reading them as text would lose their bytes.
        String preview = javaObject ? "aced0005"
                : RceCore.classifyBinary(inspected) != null
                ? java.util.Base64.getEncoder().encodeToString(java.util.Arrays.copyOf(inspected, Math.min(previewLength, 12000)))
                : new String(inspected, 0, previewLength, charset(request));
        return new Result(wrapped, preview, false);
    }

    private static Charset charset(HttpServletRequest request) {
        try {
            return request.getCharacterEncoding() == null
                    ? StandardCharsets.UTF_8 : Charset.forName(request.getCharacterEncoding());
        } catch (Exception ignored) {
            return StandardCharsets.UTF_8;
        }
    }

    @Override
    public ServletInputStream getInputStream() {
        ByteArrayInputStream input = new ByteArrayInputStream(body);
        return new ServletInputStream() {
            @Override public boolean isFinished() { return input.available() == 0; }
            @Override public boolean isReady() { return true; }
            @Override public void setReadListener(ReadListener listener) {
                if (listener == null) return;
                try {
                    if (isFinished()) listener.onAllDataRead(); else listener.onDataAvailable();
                } catch (IOException ex) {
                    listener.onError(ex);
                }
            }
            @Override public int read() { return input.read(); }
            @Override public int read(byte[] b, int off, int len) { return input.read(b, off, len); }
        };
    }

    @Override
    public BufferedReader getReader() {
        return new BufferedReader(new InputStreamReader(getInputStream(), charset(this)));
    }

    public record Result(HttpServletRequest request, String preview, boolean tooLarge) {}
}
