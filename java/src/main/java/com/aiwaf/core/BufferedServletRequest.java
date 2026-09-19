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
        BufferedServletRequest wrapped = new BufferedServletRequest(request, bytes);
        int previewLength = Math.min(bytes.length, Math.max(0, config.requestBodyInspectionBytes));
        String preview = bytes.length >= 4
                && (bytes[0] & 0xff) == 0xac && (bytes[1] & 0xff) == 0xed
                && bytes[2] == 0 && bytes[3] == 5
                ? "aced0005"
                : new String(bytes, 0, previewLength, charset(request));
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
