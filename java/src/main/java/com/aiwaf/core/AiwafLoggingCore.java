package com.aiwaf.core;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

public final class AiwafLoggingCore {
    private static final Logger logger = Logger.getLogger(AiwafLoggingCore.class.getName());

    private AiwafLoggingCore() {}

    public static void log(AiwafConfig config, AiwafRequest req, AiwafDecision decision, int statusCode, long responseTimeMs, long contentLength) {
        if (!config.loggingEnabled) return;

        File logDir = new File(config.logDir);
        if (!logDir.exists()) {
            logDir.mkdirs();
        }
        SecureFiles.restrictDirectory(logDir.toPath());

        File accessLog = new File(logDir, "access.log");
        File errorLog = new File(logDir, "error.log");
        File aiwafLog = new File(logDir, "aiwaf.log");

        String nowIso = ZonedDateTime.now(ZoneOffset.UTC).format(DateTimeFormatter.ISO_INSTANT);
        String nowCombined = ZonedDateTime.now(ZoneOffset.UTC).format(DateTimeFormatter.ofPattern("dd/MMM/yyyy:HH:mm:ss Z", Locale.ENGLISH));
        String nowSimple = ZonedDateTime.now(ZoneOffset.UTC).format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

        String ip = oneLine(req.ip(), 128);
        String method = oneLine(req.method(), 32);
        String path = oneLine(req.path(), 4096);
        String referer = oneLine(req.headers().getOrDefault("referer", "-"), 4096);
        String userAgent = oneLine(req.headers().getOrDefault("user-agent", "-"), config.maxUserAgentLength);
        boolean isBlocked = decision != null && !decision.allowed();
        String blockReason = isBlocked ? oneLine(decision.reason(), 1024) : "-";

        // Access Log
        try (FileWriter fw = new FileWriter(accessLog, true)) {
            if ("csv".equalsIgnoreCase(config.logFormat)) {
                if (accessLog.length() == 0) {
                    fw.write("timestamp,ip,method,path,query_string,protocol,status_code,content_length,response_time_ms,referer,user_agent,blocked,block_reason\n");
                }
                String qs = "-";
                if (config.logQueryParameters && req.query() != null && !req.query().isEmpty()) {
                    java.util.Map<String, String> safe = new java.util.HashMap<>();
                    req.query().forEach((k, v) -> safe.put(k,
                            config.sensitiveParameterNames.stream().anyMatch(n -> n.equalsIgnoreCase(k))
                                    ? "[redacted]" : oneLine(v, 1024)));
                    qs = safe.toString().replace(",", ";");
                }
                fw.write(String.format("%s,%s,%s,%s,%s,HTTP/1.1,%d,%d,%d,%s,%s,%s,%s\n",
                        nowIso, escapeCsv(ip), escapeCsv(method), escapeCsv(path), escapeCsv(qs),
                        statusCode, contentLength, responseTimeMs, escapeCsv(referer), escapeCsv(userAgent),
                        isBlocked ? "True" : "False", escapeCsv(blockReason)));
            } else if ("json".equalsIgnoreCase(config.logFormat)) {
                String json = String.format("{\"timestamp\":\"%s\",\"ip\":\"%s\",\"method\":\"%s\",\"path\":\"%s\",\"status_code\":%d,\"content_length\":%d,\"response_time_ms\":%d,\"referer\":\"%s\",\"user_agent\":\"%s\",\"blocked\":%b,\"block_reason\":\"%s\"}\n",
                        nowIso, jsonEscape(ip), jsonEscape(method), jsonEscape(path), statusCode, contentLength, responseTimeMs, jsonEscape(referer), jsonEscape(userAgent), isBlocked, jsonEscape(blockReason));
                fw.write(json);
            } else {
                // Combined
                String cl = contentLength >= 0 ? String.valueOf(contentLength) : "-";
                String blockStr = isBlocked ? "BLOCKED" : "-";
                String logLine = String.format("%s - - [%s] \"%s %s HTTP/1.1\" %d %s \"%s\" \"%s\" %dms %s \"%s\"\n",
                        ip, nowCombined, method, path, statusCode, cl, referer, userAgent, responseTimeMs, blockStr, blockReason);
                fw.write(logLine);
            }
        } catch (IOException e) {
            logger.log(Level.WARNING, "Failed to write access log", e);
        }
        SecureFiles.restrictFile(accessLog.toPath());

        // Error log
        if (statusCode >= 400) {
            try (FileWriter fw = new FileWriter(errorLog, true)) {
                fw.write(String.format("[%s] [error] %d from %s: %s %s \"%s\"\n", nowSimple, statusCode, ip, method, path, userAgent));
            } catch (IOException e) {
                logger.log(Level.WARNING, "Failed to write error log", e);
            }
        }

        // AIWAF log
        if (isBlocked) {
            try (FileWriter fw = new FileWriter(aiwafLog, true)) {
                fw.write(String.format("[%s] [AIWAF] BLOCKED %s - %s - %s %s \"%s\"\n", nowSimple, ip, blockReason, method, path, userAgent));
            } catch (IOException e) {
                logger.log(Level.WARNING, "Failed to write aiwaf log", e);
            }
        }
        if (errorLog.exists()) SecureFiles.restrictFile(errorLog.toPath());
        if (aiwafLog.exists()) SecureFiles.restrictFile(aiwafLog.toPath());
    }

    private static String escapeCsv(String val) {
        if (val == null) return "";
        if (!val.isEmpty() && "=+-@".indexOf(val.charAt(0)) >= 0) val = "'" + val;
        if (val.contains(",") || val.contains("\"") || val.contains("\n")) {
            return "\"" + val.replace("\"", "\"\"") + "\"";
        }
        return val;
    }

    private static String oneLine(String value, int maxLength) {
        if (value == null) return "";
        String cleaned = value.replaceAll("[\\p{Cntrl}&&[^\\t]]", " ").replace('\t', ' ');
        return cleaned.length() <= maxLength ? cleaned : cleaned.substring(0, maxLength);
    }

    private static String jsonEscape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }
}
