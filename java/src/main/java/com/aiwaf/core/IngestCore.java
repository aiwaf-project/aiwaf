package com.aiwaf.core;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

public final class IngestCore {
    private IngestCore() {}

    public static List<NormalizedEvent> eventsFromLogLines(
            List<String> lines,
            Predicate<String> knownPathFn,
            Predicate<String> exemptPathFn,
            String responseTimeUnit
    ) {
        List<NormalizedEvent> out = new ArrayList<>();
        if (lines == null) {
            return out;
        }
        for (String line : lines) {
            Map<String, Object> parsed = LogIoCore.parseLogLine(line);
            if (parsed == null) {
                continue;
            }
            String path = String.valueOf(parsed.getOrDefault("path", "/"));
            boolean known = knownPathFn != null && knownPathFn.test(path);
            boolean exempt = exemptPathFn != null && exemptPathFn.test(path);

            double rt = parsed.get("response_time") instanceof Number n ? n.doubleValue() : 0.0;
            int status = parsed.get("status") instanceof Number n ? n.intValue() : 0;
            LocalDateTime ts = parsed.get("timestamp") instanceof LocalDateTime t ? t : null;
            if (ts == null) {
                continue;
            }

            out.add(new NormalizedEvent(
                    String.valueOf(parsed.getOrDefault("ip", "")),
                    String.valueOf(parsed.getOrDefault("method", "GET")),
                    path,
                    status,
                    toMs(rt, responseTimeUnit),
                    ts,
                    String.valueOf(parsed.getOrDefault("user_agent", "")),
                    "",
                    "",
                    known,
                    exempt
            ));
        }
        return out;
    }

    public static List<NormalizedEvent> readEventsFromCsvLog(
            String path,
            Predicate<String> knownPathFn,
            Predicate<String> exemptPathFn,
            String responseTimeUnit
    ) {
        List<NormalizedEvent> events = new ArrayList<>();
        if (path == null || path.isBlank()) {
            return events;
        }
        Path csvPath = Path.of(path);
        if (!Files.exists(csvPath)) {
            return events;
        }
        try (BufferedReader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null) {
                return events;
            }
            String[] columns = header.split(",", -1);
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",", -1);
                Map<String, String> row = new HashMap<>();
                for (int i = 0; i < columns.length && i < values.length; i++) {
                    row.put(columns[i].trim(), values[i].trim());
                }

                String tsRaw = row.getOrDefault("timestamp", "");
                if (tsRaw.isBlank()) {
                    continue;
                }
                LocalDateTime ts;
                try {
                    ts = LocalDateTime.parse(tsRaw);
                } catch (DateTimeParseException ex) {
                    continue;
                }

                String reqPath = row.getOrDefault("path", "/");
                if (reqPath.isBlank()) {
                    reqPath = "/";
                }

                boolean known = knownPathFn != null ? knownPathFn.test(reqPath) : asBool(row.get("known_path"));
                boolean exempt = exemptPathFn != null ? exemptPathFn.test(reqPath) : asBool(row.get("exempt_path"));

                int status = parseInt(row.get("status_code"), parseInt(row.get("status"), 0));
                double rt = parseDouble(row.get("response_time_ms"), parseDouble(row.get("response_time"), 0.0));

                events.add(new NormalizedEvent(
                        row.getOrDefault("ip", row.getOrDefault("ip_address", "")),
                        row.getOrDefault("method", "GET"),
                        reqPath,
                        status,
                        toMs(rt, responseTimeUnit),
                        ts,
                        row.getOrDefault("user_agent", ""),
                        "",
                        "",
                        known,
                        exempt
                ));
            }
        } catch (IOException ignored) {
            return List.of();
        }
        return events;
    }

    private static double toMs(double value, String unit) {
        String normalized = unit == null ? "" : unit.toLowerCase().trim();
        if ("ms".equals(normalized) || "millisecond".equals(normalized) || "milliseconds".equals(normalized)) {
            return value;
        }
        return value * 1000.0;
    }

    private static boolean asBool(String value) {
        if (value == null) {
            return false;
        }
        String s = value.trim().toLowerCase();
        return "1".equals(s) || "true".equals(s) || "yes".equals(s) || "y".equals(s) || "on".equals(s);
    }

    private static int parseInt(String value, int fallback) {
        try {
            return Integer.parseInt(value);
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static double parseDouble(String value, double fallback) {
        try {
            return Double.parseDouble(value);
        } catch (Exception ignored) {
            return fallback;
        }
    }
}
