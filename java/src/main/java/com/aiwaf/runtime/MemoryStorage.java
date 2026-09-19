package com.aiwaf.runtime;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class MemoryStorage implements StorageBackend {
    private static final class Entry {
        Object value;
        Long expiresAtMillis;
    }

    private final Map<String, Entry> data = new ConcurrentHashMap<>();

    private void cleanupExpired() {
        long now = System.currentTimeMillis();
        for (Map.Entry<String, Entry> e : data.entrySet()) {
            Long expires = e.getValue().expiresAtMillis;
            if (expires != null && now > expires) {
                data.remove(e.getKey());
            }
        }
    }

    @Override
    public Object get(String key) {
        cleanupExpired();
        Entry entry = data.get(key);
        if (entry == null) return null;
        return entry.value;
    }

    @Override
    public boolean set(String key, Object value, Integer ttlSeconds) {
        Entry entry = new Entry();
        entry.value = value;
        entry.expiresAtMillis = ttlSeconds == null ? null : System.currentTimeMillis() + (ttlSeconds * 1000L);
        data.put(key, entry);
        return true;
    }

    @Override
    public boolean delete(String key) {
        return data.remove(key) != null;
    }

    @Override
    public boolean exists(String key) {
        return get(key) != null;
    }

    @Override
    public List<String> getAllKeys(String globPattern) {
        cleanupExpired();
        List<String> keys = new ArrayList<>();
        String pattern = globPattern == null || globPattern.isBlank() ? "*" : globPattern;
        String regex = pattern.replace(".", "\\.").replace("*", ".*");
        for (String key : data.keySet()) {
            if (key.matches(regex)) {
                keys.add(key);
            }
        }
        return keys;
    }
}
