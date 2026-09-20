package com.aiwaf.core;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Windowed, weighted UUID-tampering signals matching the Python runtime policy. */
public final class UuidScoreCore {
    public enum Signal { MALFORMED, NOT_FOUND, SUCCESS }

    public record Config(
            boolean enabled,
            int windowSeconds,
            int blockThreshold,
            int malformedWeight,
            int notFoundWeight,
            int successDecay
    ) {
        public Config {
            windowSeconds = Math.max(1, windowSeconds);
            blockThreshold = Math.max(1, blockThreshold);
        }
    }

    public record Decision(int score, boolean blocked) {}

    private record Event(long timestampMillis, int delta) {}

    private final Map<String, Deque<Event>> eventsBySubject = new ConcurrentHashMap<>();

    public Decision record(String subject, Signal signal, long nowMillis, Config config) {
        if (config == null || !config.enabled()) {
            return new Decision(0, false);
        }
        int delta = switch (signal) {
            case MALFORMED -> config.malformedWeight();
            case NOT_FOUND -> config.notFoundWeight();
            case SUCCESS -> -config.successDecay();
        };
        if (delta == 0) {
            return new Decision(0, false);
        }

        String key = subject == null || subject.isBlank() ? "unknown" : subject;
        Deque<Event> events = eventsBySubject.computeIfAbsent(key, ignored -> new ArrayDeque<>());
        long cutoff = nowMillis - config.windowSeconds() * 1_000L;
        int score;
        synchronized (events) {
            while (!events.isEmpty() && events.peekFirst().timestampMillis() < cutoff) {
                events.removeFirst();
            }
            events.addLast(new Event(nowMillis, delta));
            score = events.stream().mapToInt(Event::delta).sum();
        }
        return new Decision(score, score >= config.blockThreshold());
    }

    public void clear() {
        eventsBySubject.clear();
    }

    public void clear(String subject) {
        if (subject != null) {
            eventsBySubject.remove(subject);
        }
    }
}
