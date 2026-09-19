package com.aiwaf.core;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

public final class WhoisLookup {
    private WhoisLookup() {}

    public static Map<String, String> runWhoisLookup(String target) throws IOException {
        if (target == null || target.isBlank()) {
            throw new IllegalArgumentException("target is required");
        }
        ProcessBuilder pb = new ProcessBuilder("whois", target);
        Process proc = pb.start();
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                out.append(line).append('\n');
            }
        }
        try {
            proc.waitFor();
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
        Map<String, String> result = new HashMap<>();
        result.put("target", target);
        result.put("raw", out.toString().trim());
        return result;
    }
}
