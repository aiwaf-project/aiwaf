package com.aiwaf.runtime;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class KeywordStore {
    private final StorageBackend storage;

    public KeywordStore(StorageBackend storage) {
        this.storage = storage;
    }

    private Map<String, Integer> counts() {
        Object v = storage.get("keyword_counts");
        Map<String, Integer> out = new HashMap<>();
        if (!(v instanceof Map<?, ?> raw)) {
            return out;
        }
        for (Map.Entry<?, ?> e : raw.entrySet()) {
            if (e.getKey() == null || e.getValue() == null) {
                continue;
            }
            Integer count = null;
            if (e.getValue() instanceof Number n) {
                count = n.intValue();
            } else {
                try {
                    count = Integer.parseInt(String.valueOf(e.getValue()));
                } catch (NumberFormatException ignored) {
                    count = null;
                }
            }
            if (count != null) {
                out.put(String.valueOf(e.getKey()), count);
            }
        }
        return out;
    }

    public synchronized void addKeyword(String keyword, int count) {
        if (keyword == null || keyword.isBlank()) return;
        Map<String, Integer> counts = counts();
        String k = keyword.trim().toLowerCase();
        counts.put(k, counts.getOrDefault(k, 0) + Math.max(1, count));
        storage.set("keyword_counts", counts, null);
    }

    public synchronized void removeKeyword(String keyword) {
        if (keyword == null || keyword.isBlank()) return;
        Map<String, Integer> counts = counts();
        counts.remove(keyword.trim().toLowerCase());
        storage.set("keyword_counts", counts, null);
    }

    public synchronized List<String> getTopKeywords(int n) {
        Map<String, Integer> counts = counts();
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(counts.entrySet());
        entries.sort(Comparator.comparingInt((Map.Entry<String, Integer> e) -> e.getValue()).reversed());
        List<String> out = new ArrayList<>();
        for (Map.Entry<String, Integer> e : entries) {
            out.add(e.getKey());
            if (out.size() >= Math.max(1, n)) break;
        }
        return out;
    }
}
