package com.aiwaf.core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

public final class LogIoCore {
    private static final Pattern LOG_RX = Pattern.compile(
            "(\\d+\\.\\d+\\.\\d+\\.\\d+).*\\[(.*?)\\].*\"(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) (.*?) HTTP/.*?\" "
                    + "(\\d{3}).*?\"(.*?)\" \"(.*?)\""
    );
    private static final Pattern RESPONSE_TIME_RX = Pattern.compile("response-time=(\\d+\\.\\d+)");
    private static final DateTimeFormatter LOG_TS = DateTimeFormatter.ofPattern("dd/MMM/yyyy:HH:mm:ss", Locale.ENGLISH);

    private LogIoCore() {}

    public static List<String> readRotatedLogs(String basePath) {
        List<String> lines = new ArrayList<>();
        if (basePath == null || basePath.isBlank()) {
            return lines;
        }
        Path base = Path.of(basePath);
        if (Files.exists(base)) {
            lines.addAll(readTextLines(base));
        }

        Path parent = base.getParent() == null ? Path.of(".") : base.getParent();
        String prefix = base.getFileName().toString() + ".";
        List<Path> rotated = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(parent)) {
            for (Path candidate : stream) {
                String name = candidate.getFileName().toString();
                if (name.startsWith(prefix)) {
                    rotated.add(candidate);
                }
            }
        } catch (IOException ignored) {
            return lines;
        }
        rotated.sort(Comparator.comparing(Path::toString));
        for (Path path : rotated) {
            if (path.getFileName().toString().endsWith(".gz")) {
                lines.addAll(readGzipLines(path));
            } else {
                lines.addAll(readTextLines(path));
            }
        }
        return lines;
    }

    public static Map<String, Object> parseLogLine(String line) {
        if (line == null || line.isBlank()) {
            return null;
        }
        Matcher matcher = LOG_RX.matcher(line);
        if (!matcher.find()) {
            return null;
        }
        String ip = matcher.group(1);
        String tsRaw = matcher.group(2);
        String method = matcher.group(3);
        String path = matcher.group(4);
        String status = matcher.group(5);
        String referer = matcher.group(6);
        String userAgent = matcher.group(7);

        LocalDateTime ts;
        try {
            String main = tsRaw.split("\\s+")[0];
            ts = LocalDateTime.parse(main, LOG_TS);
        } catch (DateTimeParseException ex) {
            return null;
        }

        double responseTime = 0.0;
        Matcher rt = RESPONSE_TIME_RX.matcher(line);
        if (rt.find()) {
            try {
                responseTime = Double.parseDouble(rt.group(1));
            } catch (NumberFormatException ignored) {
                responseTime = 0.0;
            }
        }

        Map<String, Object> out = new HashMap<>();
        out.put("ip", ip);
        out.put("timestamp", ts);
        out.put("method", method);
        out.put("path", path);
        out.put("status", Integer.parseInt(status));
        out.put("referer", referer);
        out.put("user_agent", userAgent);
        out.put("response_time", responseTime);
        return out;
    }

    public static void writeCsvLog(String path, Map<String, Object> row, List<String> fieldNames) {
        if (path == null || path.isBlank() || row == null || fieldNames == null || fieldNames.isEmpty()) {
            return;
        }
        Path target = Path.of(path);
        Path parent = target.getParent();
        try {
            if (parent != null) {
                Files.createDirectories(parent);
            }
        } catch (IOException ignored) {
            return;
        }

        Path lockPath = Path.of(path + ".lock");
        try (FileChannel channel = FileChannel.open(
                lockPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE
        );
             FileLock ignored = channel.lock()) {
            boolean needsHeader = !Files.exists(target) || Files.size(target) == 0;
            try (BufferedWriter writer = Files.newBufferedWriter(
                    target,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND
            )) {
                if (needsHeader) {
                    writer.write(joinCsvRow(fieldNames));
                    writer.newLine();
                }
                List<String> values = new ArrayList<>();
                for (String field : fieldNames) {
                    Object value = row.get(field);
                    values.add(value == null ? "" : String.valueOf(value));
                }
                writer.write(joinCsvRow(values));
                writer.newLine();
            }
        } catch (IOException ignored) {
        }
    }

    private static List<String> readTextLines(Path path) {
        try {
            return Files.readAllLines(path, StandardCharsets.UTF_8);
        } catch (IOException ignored) {
            return List.of();
        }
    }

    private static List<String> readGzipLines(Path path) {
        List<String> lines = new ArrayList<>();
        try (GZIPInputStream gzip = new GZIPInputStream(Files.newInputStream(path));
             BufferedReader reader = new BufferedReader(new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException ignored) {
            return List.of();
        }
        return lines;
    }

    private static String joinCsvRow(List<String> values) {
        List<String> escaped = new ArrayList<>();
        for (String value : values) {
            String v = value == null ? "" : value;
            boolean needsQuotes = v.contains(",") || v.contains("\"") || v.contains("\n") || v.contains("\r");
            if (v.contains("\"")) {
                v = v.replace("\"", "\"\"");
            }
            if (needsQuotes) {
                v = "\"" + v + "\"";
            }
            escaped.add(v);
        }
        return String.join(",", escaped);
    }
}
