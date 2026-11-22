package com.aiwaf.core;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * WebSocket messages, checked like request bodies. HTTP checks never see what a WebSocket carries
 * after its handshake, so a message handler calls {@link #check} with each message a client
 * sends: SQL injection, remote code execution and their disguises, and JSON messages by their
 * values. For Spring, in a {@code TextWebSocketHandler}:
 *
 * <pre>{@code
 * String rule = AiwafMessageCore.check(message.getPayload(), config, session.getUri().getPath());
 * if (rule != null) { session.close(CloseStatus.POLICY_VIOLATION); return; }
 * }</pre>
 *
 * <p>It follows {@code sqlInjectionMode}: "monitor" logs the rule and returns null, "off" checks
 * nothing. Log events carry the rule and path only, never message content.
 */
public final class AiwafMessageCore {
    private static final Logger LOGGER = Logger.getLogger("com.aiwaf.websocket");
    private static final ObjectMapper JSON = new ObjectMapper();
    /** The close code for a message with a finding: 1008, policy violation. */
    public static final int POLICY_VIOLATION = 1008;
    /** The close code for a message too large to check: 1009. */
    public static final int TOO_LARGE = 1009;

    private AiwafMessageCore() {}

    /** The rule for one message, whatever the mode, or null. */
    public static String inspect(String message, AiwafConfig config) {
        if (message == null || message.isEmpty()) return null;
        int budget = config.requestBodyInspectionBytes;
        if (message.getBytes(StandardCharsets.UTF_8).length > budget) return "payload_inspection_limit";
        return SqlInjectionCore.inspectQueryAndBody(null, message, budget);
    }

    /** A binary message, checked for serialized objects and as UTF-8 text. */
    public static String inspect(byte[] message, AiwafConfig config) {
        if (message == null || message.length == 0) return null;
        String binary = RceCore.classifyBinary(message);
        return binary != null ? binary : inspect(new String(message, StandardCharsets.UTF_8), config);
    }

    /** The rule to close the socket for, after logging it, or null: null in monitor and off modes. */
    public static String check(Object message, AiwafConfig config, String path) {
        String mode = config.sqlInjectionMode == null ? "block" : config.sqlInjectionMode;
        if ("off".equals(mode)) return null;
        String rule = message instanceof byte[] bytes ? inspect(bytes, config) : inspect(message == null ? null : String.valueOf(message), config);
        if (rule == null) return null;
        Map<String, String> line = new LinkedHashMap<>();
        line.put("event", "aiwaf.websocket");
        line.put("mode", mode);
        line.put("path", path == null ? "" : path);
        line.put("rule", rule);
        try {
            LOGGER.warning(JSON.writeValueAsString(line));
        } catch (Exception ignored) {
            // Logging must not decide the outcome.
        }
        return "monitor".equals(mode) ? null : rule;
    }

    /** The close code for a rule {@link #check} returned. */
    public static int closeCode(String rule) {
        return "payload_inspection_limit".equals(rule) ? TOO_LARGE : POLICY_VIOLATION;
    }
}
