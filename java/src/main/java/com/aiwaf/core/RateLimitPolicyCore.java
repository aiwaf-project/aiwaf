package com.aiwaf.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class RateLimitPolicyCore {
    private RateLimitPolicyCore() {}

    public record RateLimitConfig(int maxRequestsPerWindow, int windowSeconds, int floodBurstThreshold) {
        public RateLimitConfig() {
            this(20, 10, 40);
        }
    }

    public record RateLimitDecision(
            boolean allow,
            boolean softLimited,
            boolean hardBlocked,
            Map<String, Integer> metadata
    ) {}

    public record RequestKey(String ip, String path) {}

    public static final class RateLimiter {
        private final RateLimitConfig config;
        private final Map<String, List<Double>> seen = new HashMap<>();
        private final Map<String, List<Double>> pathSeen = new HashMap<>();
        private final Map<String, Double> lastSeen = new HashMap<>();

        public RateLimiter(RateLimitConfig config) {
            this.config = config == null ? new RateLimitConfig() : config;
        }

        public synchronized RateLimitDecision registerWithSoftHardLimits(
                RequestKey request,
                double nowTs,
                Integer softLimit,
                Integer hardLimit,
                Integer windowSeconds
        ) {
            int window = windowSeconds == null ? config.windowSeconds() : windowSeconds;
            int soft = softLimit == null ? config.maxRequestsPerWindow() : softLimit;
            int hard = hardLimit == null ? config.floodBurstThreshold() : hardLimit;

            if (soft < 0) soft = 0;
            if (hard < 0) hard = 0;
            if (hard < soft) hard = soft;

            String ip = request == null || request.ip() == null ? "" : request.ip();
            List<Double> bucket = seen.computeIfAbsent(ip, ignored -> new ArrayList<>());
            double cutoff = nowTs - window;
            bucket.removeIf(ts -> ts < cutoff);
            bucket.add(nowTs);

            int count = bucket.size();
            boolean softLimited = count > soft;
            boolean hardBlocked = count > hard;

            Map<String, Integer> meta = new HashMap<>();
            meta.put("count", count);
            meta.put("window_seconds", window);
            meta.put("soft_limit", soft);
            meta.put("hard_limit", hard);
            meta.put("soft_limited", softLimited ? 1 : 0);
            meta.put("hard_blocked", hardBlocked ? 1 : 0);

            return new RateLimitDecision(!hardBlocked, softLimited, hardBlocked, meta);
        }

        public synchronized RegisterResult register(RequestKey request, double nowTs) {
            String ip = request == null || request.ip() == null ? "" : request.ip();
            String path = request == null || request.path() == null ? "/" : request.path();
            List<Double> bucket = seen.computeIfAbsent(ip, ignored -> new ArrayList<>());
            double cutoff = nowTs - config.windowSeconds();
            bucket.removeIf(ts -> ts < cutoff);
            bucket.add(nowTs);

            String pathKey = ip + "|" + path;
            List<Double> pathBucket = pathSeen.computeIfAbsent(pathKey, ignored -> new ArrayList<>());
            pathBucket.removeIf(ts -> ts < cutoff);
            pathBucket.add(nowTs);

            int count = bucket.size();
            boolean flood = count >= config.floodBurstThreshold();
            boolean limited = count > config.maxRequestsPerWindow();
            boolean pathFlood = pathBucket.size() >= Math.max(2, config.floodBurstThreshold() / 2);

            int rapid = 0;
            Double last = lastSeen.get(ip);
            if (last != null) {
                rapid = (nowTs - last) < 0.05 ? 1 : 0;
            }
            lastSeen.put(ip, nowTs);

            Map<String, Integer> meta = new HashMap<>();
            meta.put("count", count);
            meta.put("flood", flood ? 1 : 0);
            meta.put("limited", limited ? 1 : 0);
            meta.put("path_flood", pathFlood ? 1 : 0);
            meta.put("rapid_repeat", rapid);
            return new RegisterResult(!limited, meta);
        }
    }

    public static RateLimitWindowResult applyRateLimitWindow(
            List<Double> timestamps,
            double nowTs,
            int windowSeconds,
            int softLimit,
            int hardLimit
    ) {
        int window = windowSeconds <= 0 ? 1 : windowSeconds;
        int soft = Math.max(0, softLimit);
        int hard = Math.max(0, hardLimit);
        if (hard < soft) hard = soft;

        List<Double> kept = new ArrayList<>();
        if (timestamps != null) {
            for (Double ts : timestamps) {
                if (ts != null && (nowTs - ts) < window) {
                    kept.add(ts);
                }
            }
        }
        kept.add(nowTs);

        int count = kept.size();
        boolean softLimited = count > soft;
        boolean hardBlocked = count > hard;

        Map<String, Integer> meta = new HashMap<>();
        meta.put("count", count);
        meta.put("window_seconds", window);
        meta.put("soft_limit", soft);
        meta.put("hard_limit", hard);
        meta.put("soft_limited", softLimited ? 1 : 0);
        meta.put("hard_blocked", hardBlocked ? 1 : 0);

        return new RateLimitWindowResult(
                kept,
                new RateLimitDecision(!hardBlocked, softLimited, hardBlocked, meta)
        );
    }

    public record RateLimitWindowResult(List<Double> kept, RateLimitDecision decision) {}
    public record RegisterResult(boolean ok, Map<String, Integer> metadata) {}
}
