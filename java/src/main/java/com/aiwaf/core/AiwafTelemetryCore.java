package com.aiwaf.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;

public final class AiwafTelemetryCore {
    private static final long[] LATENCY_BUCKETS_MS = new long[]{1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000};
    private final ConcurrentMap<String, AtomicLong> counters = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, Histogram> histograms = new ConcurrentHashMap<>();

    public void increment(String key) {
        counters.computeIfAbsent(key, k -> new AtomicLong()).incrementAndGet();
    }

    public void add(String key, long delta) {
        counters.computeIfAbsent(key, k -> new AtomicLong()).addAndGet(delta);
    }

    public void observeMs(String key, long valueMs) {
        histograms.computeIfAbsent(key, k -> new Histogram()).observe(valueMs);
    }

    public Map<String, Long> counterSnapshot() {
        Map<String, Long> out = new java.util.HashMap<>();
        for (Map.Entry<String, AtomicLong> e : counters.entrySet()) {
            out.put(e.getKey(), e.getValue().get());
        }
        return Collections.unmodifiableMap(out);
    }

    public Map<String, HistogramSnapshot> histogramSnapshot() {
        Map<String, HistogramSnapshot> out = new java.util.HashMap<>();
        for (Map.Entry<String, Histogram> e : histograms.entrySet()) {
            out.put(e.getKey(), e.getValue().snapshot());
        }
        return Collections.unmodifiableMap(out);
    }

    public void reset() {
        counters.clear();
        histograms.clear();
    }

    private static final class Histogram {
        private final AtomicLong count = new AtomicLong();
        private final AtomicLong sum = new AtomicLong();
        private final AtomicLong min = new AtomicLong(Long.MAX_VALUE);
        private final AtomicLong max = new AtomicLong(Long.MIN_VALUE);
        private final List<AtomicLong> buckets = initBuckets();

        void observe(long value) {
            long v = Math.max(0, value);
            count.incrementAndGet();
            sum.addAndGet(v);
            updateMin(v);
            updateMax(v);
            int idx = bucketIndex(v);
            buckets.get(idx).incrementAndGet();
        }

        HistogramSnapshot snapshot() {
            long c = count.get();
            long s = sum.get();
            long minV = c == 0 ? 0 : min.get();
            long maxV = c == 0 ? 0 : max.get();
            double avg = c == 0 ? 0.0 : (double) s / c;
            List<Long> bucketCounts = new ArrayList<>(buckets.size());
            for (AtomicLong b : buckets) {
                bucketCounts.add(b.get());
            }
            return new HistogramSnapshot(c, minV, maxV, avg, LATENCY_BUCKETS_MS, bucketCounts);
        }

        private static List<AtomicLong> initBuckets() {
            List<AtomicLong> out = new ArrayList<>();
            for (int i = 0; i < LATENCY_BUCKETS_MS.length + 1; i++) {
                out.add(new AtomicLong());
            }
            return out;
        }

        private static int bucketIndex(long v) {
            for (int i = 0; i < LATENCY_BUCKETS_MS.length; i++) {
                if (v <= LATENCY_BUCKETS_MS[i]) return i;
            }
            return LATENCY_BUCKETS_MS.length;
        }

        private void updateMin(long v) {
            long curr;
            do {
                curr = min.get();
                if (v >= curr) return;
            } while (!min.compareAndSet(curr, v));
        }

        private void updateMax(long v) {
            long curr;
            do {
                curr = max.get();
                if (v <= curr) return;
            } while (!max.compareAndSet(curr, v));
        }
    }

    public record HistogramSnapshot(
            long count,
            long minMs,
            long maxMs,
            double avgMs,
            long[] upperBoundsMs,
            List<Long> bucketCounts
    ) {}
}
