package com.aiwaf.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class FeaturesCore {
    private static final Pattern TOKEN_RE = Pattern.compile("[a-zA-Z0-9_\\-]{2,64}");
    private static final Map<String, Integer> STATUS_IDX = Map.of("200", 0, "403", 1, "404", 2, "500", 3);
    static final List<String> PYTHON_FEATURE_NAMES = List.of(
            "path_len", "kw_hits", "resp_time", "status_idx", "burst_count", "total_404"
    );

    private FeaturesCore() {}

    public static List<FeatureVectorCore> extractFeatures(
            List<NormalizedEvent> events,
            Set<String> staticKeywords
    ) {
        Map<String, Integer> ip404 = new HashMap<>();
        Map<String, List<Double>> ipTimes = new HashMap<>();
        for (NormalizedEvent event : events) {
            ipTimes.computeIfAbsent(event.ip(), k -> new ArrayList<>()).add(epochSeconds(event));
            if (event.statusCode() == 404 && !event.exemptPath()) {
                ip404.put(event.ip(), ip404.getOrDefault(event.ip(), 0) + 1);
            }
        }
        for (List<Double> times : ipTimes.values()) {
            Collections.sort(times);
        }

        List<FeatureVectorCore> vectors = new ArrayList<>();
        for (NormalizedEvent event : events) {
            double ts = epochSeconds(event);
            List<Double> times = ipTimes.getOrDefault(event.ip(), List.of());
            int burst = upperBound(times, ts) - lowerBound(times, ts - 10.0);
            int kwHits = 0;
            if (!event.knownPath() && !event.exemptPath()) {
                String pathLower = event.path().toLowerCase(Locale.ROOT);
                for (String kw : staticKeywords) {
                    if (pathLower.contains(kw.toLowerCase(Locale.ROOT))) {
                        kwHits++;
                    }
                }
            }
            Map<String, Double> values = new HashMap<>();
            values.put("path_len", (double) event.path().codePointCount(0, event.path().length()));
            values.put("kw_hits", (double) kwHits);
            values.put("status_idx", (double) STATUS_IDX.getOrDefault(String.valueOf(event.statusCode()), -1));
            values.put("resp_time", event.responseTimeMs() / 1000.0);
            values.put("burst_count", (double) burst);
            values.put("total_404", (double) ip404.getOrDefault(event.ip(), 0));
            vectors.add(new FeatureVectorCore(values));
        }
        return vectors;
    }

    private static double epochSeconds(NormalizedEvent event) {
        var instant = event.timestamp().atZone(java.time.ZoneId.systemDefault()).toInstant();
        return instant.getEpochSecond() + instant.getNano() / 1_000_000_000.0;
    }

    private static int lowerBound(List<Double> values, double target) {
        int low = 0;
        int high = values.size();
        while (low < high) {
            int mid = (low + high) >>> 1;
            if (values.get(mid) < target) low = mid + 1;
            else high = mid;
        }
        return low;
    }

    private static int upperBound(List<Double> values, double target) {
        int low = 0;
        int high = values.size();
        while (low < high) {
            int mid = (low + high) >>> 1;
            if (values.get(mid) <= target) low = mid + 1;
            else high = mid;
        }
        return low;
    }

    public static List<String> extractKeywordsFromEvents(
            List<NormalizedEvent> events,
            KeywordLearningConfig config
    ) {
        Map<String, Integer> counter = new HashMap<>();
        Set<String> stopwords = new HashSet<>(config.stopwords());

        for (NormalizedEvent event : events) {
            if (event.knownPath() || event.exemptPath()) continue;
            if (event.statusCode() != 404) continue;
            if (!AnomalyCore.isScanningPath(event.path())) continue;

            Matcher matcher = TOKEN_RE.matcher(event.path().toLowerCase(Locale.ROOT));
            while (matcher.find()) {
                String token = matcher.group();
                if (token.length() < config.minTokenLength() || token.length() > config.maxTokenLength()) continue;
                if (stopwords.contains(token)) continue;
                counter.put(token, counter.getOrDefault(token, 0) + 1);
            }
        }

        List<String> out = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : counter.entrySet()) {
            if (entry.getValue() >= config.minOccurrenceToLearn()) {
                out.add(entry.getKey());
            }
        }
        out.sort(String::compareTo);
        return out;
    }

    public record KeywordLearningConfig(
            int minTokenLength,
            int maxTokenLength,
            int minOccurrenceToLearn,
            List<String> stopwords
    ) {
        public KeywordLearningConfig() {
            this(4, 64, 2, List.of("http", "https", "www", "api", "json", "html"));
        }
    }
}
