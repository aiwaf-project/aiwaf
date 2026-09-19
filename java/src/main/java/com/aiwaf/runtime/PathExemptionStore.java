package com.aiwaf.runtime;

import com.aiwaf.core.ExemptionsCore;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class PathExemptionStore {
    private final StorageBackend storage;
    private final Set<String> exemptPaths = new HashSet<>();

    public PathExemptionStore(StorageBackend storage) {
        this.storage = storage;
        load();
    }

    private void load() {
        Object v = storage.get("path_exemptions");
        if (!(v instanceof java.util.Map<?, ?> map)) {
            return;
        }
        Object paths = map.get("paths");
        if (paths instanceof List<?> list) {
            for (Object o : list) {
                if (o != null && !String.valueOf(o).isBlank()) {
                    exemptPaths.add(String.valueOf(o));
                }
            }
        }
    }

    private void save() {
        java.util.Map<String, Object> m = new java.util.HashMap<>();
        m.put("paths", new ArrayList<>(exemptPaths));
        storage.set("path_exemptions", m, null);
    }

    public synchronized boolean addPath(String path, String reason) {
        if (path == null || path.isBlank()) {
            return false;
        }
        boolean added = exemptPaths.add(path.trim());
        if (added) {
            save();
        }
        return added;
    }

    public synchronized boolean removePath(String path) {
        if (path == null || path.isBlank()) {
            return false;
        }
        boolean changed = exemptPaths.remove(path.trim());
        if (changed) {
            save();
        }
        return changed;
    }

    public synchronized Set<String> getPaths() {
        return new HashSet<>(exemptPaths);
    }

    public synchronized boolean isExempted(String path, boolean allowWildcards, boolean allowPrefix) {
        return ExemptionsCore.isPathExempt(path, exemptPaths, allowWildcards, allowPrefix);
    }
}
