package com.aiwaf.runtime;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class ExemptionStore {
    private final StorageBackend storage;
    private final Set<String> exemptIps = new HashSet<>();
    private final List<String> exemptPatterns = new ArrayList<>();

    public ExemptionStore(StorageBackend storage) {
        this.storage = storage;
        load();
    }

    private void load() {
        exemptIps.clear();
        exemptPatterns.clear();
        Object v = storage.get("exemptions");
        if (!(v instanceof java.util.Map<?, ?>)) return;
        java.util.Map<?, ?> m = (java.util.Map<?, ?>) v;
        Object ips = m.get("ips");
        Object patterns = m.get("patterns");
        if (ips instanceof List<?>) {
            List<?> ipsList = (List<?>) ips;
            for (Object o : ipsList) if (o != null) exemptIps.add(String.valueOf(o));
        }
        if (patterns instanceof List<?>) {
            List<?> patternList = (List<?>) patterns;
            for (Object o : patternList) if (o != null) exemptPatterns.add(String.valueOf(o));
        }
    }

    private void save() {
        java.util.Map<String, Object> m = new java.util.HashMap<>();
        m.put("ips", new ArrayList<>(exemptIps));
        m.put("patterns", new ArrayList<>(exemptPatterns));
        storage.set("exemptions", m, null);
    }

    public synchronized void addIp(String ip, String reason) {
        load();
        exemptIps.add(ip);
        save();
    }

    public synchronized boolean removeIp(String ip) {
        load();
        boolean changed = exemptIps.remove(ip);
        if (changed) save();
        return changed;
    }

    public synchronized void addPattern(String pattern, String reason) {
        load();
        if (!exemptPatterns.contains(pattern)) {
            exemptPatterns.add(pattern);
            save();
        }
    }

    public synchronized boolean removePattern(String pattern) {
        load();
        boolean changed = exemptPatterns.remove(pattern);
        if (changed) save();
        return changed;
    }

    public synchronized boolean isExempted(String ip) {
        load();
        if (ip == null || ip.isBlank()) return false;
        if (exemptIps.contains(ip)) return true;
        for (String pattern : exemptPatterns) {
            if (pattern == null || pattern.isBlank()) continue;
            String regex = pattern.replace(".", "\\.").replace("*", ".*");
            if (ip.matches(regex)) return true;
            if (pattern.contains("/")) {
                if (CidrUtil.contains(pattern, ip)) return true;
            }
        }
        return false;
    }

    public synchronized Set<String> getExemptedIps() {
        load();
        return new HashSet<>(exemptIps);
    }

    public synchronized List<String> getExemptedPatterns() {
        load();
        return new ArrayList<>(exemptPatterns);
    }
}
