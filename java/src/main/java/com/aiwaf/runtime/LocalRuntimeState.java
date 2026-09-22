package com.aiwaf.runtime;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Per-engine bounded runtime state for non-distributed storage backends. */
public final class LocalRuntimeState implements RuntimeState {
    private record UuidEvent(long timestampMillis, int delta) {}
    private record RecentEvent(long timestampMillis, int statusCode) {}

    private final Map<String, Deque<Long>> requestBuckets = new ConcurrentHashMap<>();
    private final Map<String, Long> formGets = new ConcurrentHashMap<>();
    private final Map<String, Deque<UuidEvent>> uuidEvents = new ConcurrentHashMap<>();
    private final Map<String, Deque<RecentEvent>> recentEvents = new ConcurrentHashMap<>();

    @Override
    public RateResult recordRate(
            String key,
            long nowMillis,
            int windowSeconds,
            int maxRequests,
            int floodThreshold,
            int maxEntries
    ) {
        long cutoff = nowMillis - Math.max(1, windowSeconds) * 1_000L;
        if (!reserveDeque(requestBuckets, key, cutoff, maxEntries)) {
            return new RateResult(RateAction.CAPACITY, 0);
        }
        Deque<Long> bucket = requestBuckets.computeIfAbsent(key, ignored -> new ArrayDeque<>());
        synchronized (bucket) {
            while (!bucket.isEmpty() && bucket.peekFirst() < cutoff) bucket.removeFirst();
            bucket.addLast(nowMillis);
            int count = bucket.size();
            if (count > Math.max(1, floodThreshold)) return new RateResult(RateAction.FLOOD, count);
            if (count > Math.max(1, maxRequests)) return new RateResult(RateAction.LIMIT, count);
            return new RateResult(RateAction.ALLOW, count);
        }
    }

    @Override
    public boolean recordFormGet(String key, long nowMillis, int ttlSeconds, int maxEntries) {
        if (!formGets.containsKey(key) && formGets.size() >= bounded(maxEntries)) {
            long cutoff = nowMillis - Math.max(1, ttlSeconds) * 1_000L;
            formGets.entrySet().removeIf(entry -> entry.getValue() < cutoff);
            if (formGets.size() >= bounded(maxEntries)) return false;
        }
        formGets.put(key, nowMillis);
        return true;
    }

    @Override
    public Long getFormGet(String key) {
        return formGets.get(key);
    }

    @Override
    public UuidResult recordUuid(
            String subject,
            int delta,
            long nowMillis,
            int windowSeconds,
            int blockThreshold,
            int maxEntries
    ) {
        if (delta == 0) return new UuidResult(0, false);
        long cutoff = nowMillis - Math.max(1, windowSeconds) * 1_000L;
        if (!reserveDeque(uuidEvents, subject, cutoff, maxEntries)) return new UuidResult(0, false);
        Deque<UuidEvent> events = uuidEvents.computeIfAbsent(subject, ignored -> new ArrayDeque<>());
        synchronized (events) {
            while (!events.isEmpty() && events.peekFirst().timestampMillis() < cutoff) events.removeFirst();
            events.addLast(new UuidEvent(nowMillis, delta));
            int score = events.stream().mapToInt(UuidEvent::delta).sum();
            return new UuidResult(score, score >= Math.max(1, blockThreshold));
        }
    }

    @Override
    public void recordRecent(
            String subject,
            long nowMillis,
            int statusCode,
            int windowSeconds,
            int maxEntries
    ) {
        long cutoff = nowMillis - Math.max(1, windowSeconds) * 1_000L;
        if (!reserveDeque(recentEvents, subject, cutoff, maxEntries)) return;
        Deque<RecentEvent> events = recentEvents.computeIfAbsent(subject, ignored -> new ArrayDeque<>());
        synchronized (events) {
            events.addLast(new RecentEvent(nowMillis, statusCode));
            while (!events.isEmpty() && events.peekFirst().timestampMillis() < cutoff) events.removeFirst();
        }
    }

    @Override
    public RecentStats recentStats(String subject, long nowMillis, int windowSeconds) {
        Deque<RecentEvent> events = recentEvents.get(subject);
        if (events == null) return RecentStats.EMPTY;
        long cutoff = nowMillis - Math.max(1, windowSeconds) * 1_000L;
        long burstCutoff = nowMillis - 10_000L;
        int count = 0;
        int notFound = 0;
        int burst = 0;
        synchronized (events) {
            while (!events.isEmpty() && events.peekFirst().timestampMillis() < cutoff) events.removeFirst();
            for (RecentEvent event : events) {
                if (event.timestampMillis() >= cutoff) {
                    count++;
                    if (event.statusCode() == 404) notFound++;
                    if (event.timestampMillis() >= burstCutoff) burst++;
                }
            }
        }
        return new RecentStats(count, notFound, burst);
    }

    @Override
    public void clear() {
        requestBuckets.clear();
        formGets.clear();
        uuidEvents.clear();
        recentEvents.clear();
    }

    private static int bounded(int maxEntries) {
        return Math.max(100, maxEntries);
    }

    private static <T> boolean reserveDeque(
            Map<String, Deque<T>> state,
            String key,
            long cutoff,
            int maxEntries
    ) {
        if (state.containsKey(key) || state.size() < bounded(maxEntries)) return true;
        state.entrySet().removeIf(entry -> {
            Object newest = entry.getValue().peekLast();
            if (newest == null) return true;
            long timestamp = newest instanceof Long value ? value
                    : newest instanceof UuidEvent value ? value.timestampMillis()
                    : ((RecentEvent) newest).timestampMillis();
            return timestamp < cutoff;
        });
        return state.size() < bounded(maxEntries);
    }
}
