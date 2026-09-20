package com.aiwaf.core;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Python-compatible IP reputation scoring and progressive temporary block policy. */
public final class ReputationCore {
    public static final int DEFAULT_REASON_WEIGHT = 10;
    public static final int BLOCK_THRESHOLD = 60;
    public static final int LONG_BLOCK_THRESHOLD = 80;
    public static final int FIRST_BLOCK_SECONDS = 15 * 60;
    public static final int SECOND_BLOCK_SECONDS = 60 * 60;
    public static final int REPEATED_BLOCK_SECONDS = 24 * 60 * 60;

    private static final Map<String, Integer> REASON_WEIGHTS = reasonWeights();

    private ReputationCore() {}

    public static String normalizeReason(String reason) {
        String value = reason == null ? "unknown" : reason.trim();
        return value.isEmpty() ? "unknown" : value;
    }

    public static int reasonWeight(String reason) {
        String normalized = normalizeReason(reason).toLowerCase(Locale.ROOT);
        for (Map.Entry<String, Integer> entry : REASON_WEIGHTS.entrySet()) {
            if (normalized.contains(entry.getKey())) return entry.getValue();
        }
        return DEFAULT_REASON_WEIGHT;
    }

    public static Integer progressiveDuration(int score, int offenses) {
        if (score < BLOCK_THRESHOLD) return null;
        if (score >= LONG_BLOCK_THRESHOLD || offenses >= 3) return REPEATED_BLOCK_SECONDS;
        if (offenses == 2) return SECOND_BLOCK_SECONDS;
        return FIRST_BLOCK_SECONDS;
    }

    public static Decision evaluate(Map<String, Object> existing, String reason, double nowEpochSeconds) {
        Map<String, Object> previous = existing == null ? Map.of() : existing;
        int previousScore = integer(previous.get("score"), 0);
        int previousOffenses = integer(previous.get("offenses"), 0);
        List<String> existingReasons = reasons(previous);
        existingReasons.add(normalizeReason(reason));

        Set<String> keys = new LinkedHashSet<>();
        List<String> unique = new ArrayList<>();
        for (String value : existingReasons) {
            String normalized = normalizeReason(value);
            if (keys.add(normalized.toLowerCase(Locale.ROOT))) unique.add(normalized);
        }
        int offenses = previousOffenses + 1;
        int score = Math.min(100, previousScore + reasonWeight(reason));
        Integer duration = progressiveDuration(score, offenses);
        Double expiresAt = duration == null ? null : nowEpochSeconds + duration;
        return new Decision(score, offenses, List.copyOf(unique), score >= BLOCK_THRESHOLD, duration, expiresAt);
    }

    public static String formatBlockReason(Decision decision) {
        return String.join(", ", decision.reasons())
                + "; score=" + decision.score() + "; offenses=" + decision.offenses();
    }

    private static Map<String, Integer> reasonWeights() {
        Map<String, Integer> weights = new LinkedHashMap<>();
        weights.put("scanner", 20);
        weights.put("scan", 20);
        weights.put("sqli", 40);
        weights.put("sql injection", 40);
        weights.put("xss", 30);
        weights.put("bruteforce", 25);
        weights.put("brute force", 25);
        weights.put("flood", 25);
        weights.put("rate limit", 20);
        weights.put("honeypot", 30);
        weights.put("uuid", 25);
        weights.put("header", 15);
        weights.put("geo", 20);
        weights.put("keyword", 20);
        return Map.copyOf(weights);
    }

    private static int integer(Object value, int fallback) {
        if (value instanceof Number number) return number.intValue();
        if (value != null) {
            try {
                return Integer.parseInt(String.valueOf(value));
            } catch (NumberFormatException ignored) {
                // Fall through.
            }
        }
        return fallback;
    }

    private static List<String> reasons(Map<String, Object> existing) {
        Object raw = existing.get("reasons");
        if (raw == null) raw = existing.get("reason");
        List<String> out = new ArrayList<>();
        if (raw instanceof Iterable<?> values) {
            for (Object value : values) if (value != null) out.add(String.valueOf(value));
        } else if (raw != null) {
            out.add(String.valueOf(raw));
        }
        return out;
    }

    public record Decision(
            int score,
            int offenses,
            List<String> reasons,
            boolean shouldBlock,
            Integer durationSeconds,
            Double expiresAt
    ) {}
}
