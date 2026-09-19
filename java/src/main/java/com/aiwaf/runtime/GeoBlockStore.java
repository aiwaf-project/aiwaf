package com.aiwaf.runtime;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class GeoBlockStore {
    private final StorageBackend storage;

    public GeoBlockStore(StorageBackend storage) {
        this.storage = storage;
    }

    public synchronized Set<String> getCountries() {
        Object v = storage.get("geo_blocked_countries");
        Set<String> out = new HashSet<>();
        if (v instanceof List<?> values) {
            for (Object value : values) {
                if (value != null) {
                    String s = String.valueOf(value);
                    if (!s.isBlank()) {
                        out.add(s.trim().toUpperCase());
                    }
                }
            }
        }
        return out;
    }

    public synchronized void addCountry(String code) {
        if (code == null || code.isBlank()) return;
        Set<String> current = getCountries();
        current.add(code.trim().toUpperCase());
        storage.set("geo_blocked_countries", new ArrayList<>(current), null);
    }

    public synchronized void removeCountry(String code) {
        if (code == null || code.isBlank()) return;
        Set<String> current = getCountries();
        current.remove(code.trim().toUpperCase());
        storage.set("geo_blocked_countries", new ArrayList<>(current), null);
    }
}
