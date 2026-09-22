package com.aiwaf.runtime;

/**
 * Mutable request state used by enforcement rules. Implementations may be local
 * to one engine or shared by every application instance through Redis.
 */
public interface RuntimeState {
    enum RateAction { ALLOW, LIMIT, FLOOD, CAPACITY }

    record RateResult(RateAction action, int count) {}
    record UuidResult(int score, boolean blocked) {}
    record RecentStats(int count, int notFoundCount, int burstCount) {
        public static final RecentStats EMPTY = new RecentStats(0, 0, 0);
    }

    RateResult recordRate(
            String key,
            long nowMillis,
            int windowSeconds,
            int maxRequests,
            int floodThreshold,
            int maxEntries
    );

    boolean recordFormGet(String key, long nowMillis, int ttlSeconds, int maxEntries);

    Long getFormGet(String key);

    UuidResult recordUuid(
            String subject,
            int delta,
            long nowMillis,
            int windowSeconds,
            int blockThreshold,
            int maxEntries
    );

    void recordRecent(
            String subject,
            long nowMillis,
            int statusCode,
            int windowSeconds,
            int maxEntries
    );

    RecentStats recentStats(String subject, long nowMillis, int windowSeconds);

    void clear();
}
