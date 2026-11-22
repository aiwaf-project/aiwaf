package com.aiwaf.core;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.Part;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;

/**
 * A multipart form's parts, as the container parsed them, in a text form the payload check
 * reads: each part's headers, then its first bytes, one char per byte (ISO-8859-1), so a
 * binary file keeps its bytes. Only the payload check reads it; the application gets its parts
 * from the container as usual.
 */
public final class MultipartPreview {
    static final String BOUNDARY = "aiwaf-parts-3f9c1e";
    static final String START = "--" + BOUNDARY + "\r\n";

    private MultipartPreview() {}

    /**
     * The preview, or "" when the container can't give parts. The parameters are read first:
     * that parses the form only when the servlet has a multipart config, and otherwise leaves the
     * upload stream unread and records that there are no parts. getParts then returns what was
     * parsed. Calling getParts first would fail on a servlet with no config, and Tomcat would
     * keep that failure and throw it again when the application reads its parameters.
     */
    static String of(HttpServletRequest request, int budget) {
        Collection<Part> parts;
        try {
            request.getParameterMap();
            parts = request.getParts();
        } catch (Exception ex) {
            return "";
        }
        StringBuilder out = new StringBuilder();
        int count = 0;
        for (Part part : parts) {
            if (++count > 1024 || out.length() >= budget) break;
            out.append(START);
            for (String name : part.getHeaderNames()) {
                for (String value : part.getHeaders(name)) out.append(name).append(": ").append(value).append("\r\n");
            }
            out.append("\r\n");
            try (InputStream in = part.getInputStream()) {
                out.append(new String(in.readNBytes(Math.max(0, budget - out.length())), StandardCharsets.ISO_8859_1));
            } catch (Exception ex) {
                // A part that can't be read is checked by its headers.
            }
            out.append("\r\n");
        }
        if (count == 0) return "";
        return out.append("--").append(BOUNDARY).append("--\r\n").toString();
    }

    record PartView(String headers, String name, boolean file, byte[] content) {}

    static boolean is(String preview) {
        return preview != null && preview.startsWith(START);
    }

    /** The parts of a preview. */
    static List<PartView> parts(String preview) {
        List<PartView> parts = new ArrayList<>();
        String delimiter = "\r\n--" + BOUNDARY;
        int at = START.length();
        while (at < preview.length()) {
            int end = preview.indexOf(delimiter, at);
            if (end < 0) end = preview.length();
            String chunk = preview.substring(at, end);
            int split = chunk.indexOf("\r\n\r\n");
            String headers = split < 0 ? chunk : chunk.substring(0, split);
            String content = split < 0 ? "" : chunk.substring(split + 4);
            parts.add(new PartView(headers, param(headers, "name"), param(headers, "filename") != null || param(headers, "filename*") != null,
                    content.getBytes(StandardCharsets.ISO_8859_1)));
            if (end + delimiter.length() >= preview.length() || preview.startsWith("--", end + delimiter.length())) break;
            at = end + delimiter.length() + 2;
        }
        return parts;
    }

    /** A Content-Disposition parameter, as in name="a" or filename=x.txt, or null. */
    static String param(String headers, String key) {
        for (String line : headers.split("\r\n")) {
            String lower = line.toLowerCase(Locale.ROOT);
            if (!lower.startsWith("content-disposition")) continue;
            int from = 0;
            while (true) {
                int found = lower.indexOf(key, from);
                if (found < 0) return null;
                from = found + key.length();
                if (found > 0 && ";\t ".indexOf(lower.charAt(found - 1)) < 0) continue;
                String rest = line.substring(from).stripLeading();
                if (!rest.startsWith("=")) continue;
                rest = rest.substring(1).stripLeading();
                if (rest.startsWith("\"")) {
                    int close = rest.indexOf('"', 1);
                    return close < 0 ? rest.substring(1) : rest.substring(1, close);
                }
                int semi = rest.indexOf(';');
                return (semi < 0 ? rest : rest.substring(0, semi)).trim();
            }
        }
        return null;
    }
}
