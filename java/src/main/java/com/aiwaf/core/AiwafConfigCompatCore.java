package com.aiwaf.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class AiwafConfigCompatCore {
    private AiwafConfigCompatCore() {}

    public static AiwafConfig fromStructured(Map<String, Object> settings) {
        AiwafConfig cfg = new AiwafConfig();
        applyStructured(cfg, settings);
        return cfg;
    }

    public static AiwafConfig fromStructuredAndEnv(Map<String, Object> settings, Map<String, String> env) {
        AiwafConfig cfg = fromStructured(settings);
        applyEnv(cfg, env);
        return cfg;
    }

    public static void applyStructured(AiwafConfig cfg, Map<String, Object> settings) {
        if (cfg == null || settings == null) return;

        Map<String, Object> storage = map(settings.get("storage"));
        cfg.storageBackend = str(storage.get("backend"), cfg.storageBackend);
        cfg.storageFilePath = str(storage.get("file_path"), cfg.storageFilePath);

        Map<String, Object> hv = map(settings.get("header_validation"));
        cfg.headerValidationEnabled = bool(hv.get("enabled"), cfg.headerValidationEnabled);
        cfg.minHeaderQualityScore = intval(hv.get("quality_threshold"), cfg.minHeaderQualityScore);
        cfg.exemptPaths.addAll(strSet(hv.get("exempt_paths")));

        Map<String, Object> rl = map(settings.get("rate_limiting"));
        cfg.rateLimitEnabled = bool(rl.get("enabled"), cfg.rateLimitEnabled);
        cfg.rateLimitWindowSeconds = intval(rl.get("window_seconds"), cfg.rateLimitWindowSeconds);
        cfg.rateLimitMax = intval(rl.get("max_requests"), cfg.rateLimitMax);
        cfg.rateLimitFloodThreshold = intval(rl.get("flood_threshold"), cfg.rateLimitFloodThreshold);

        Map<String, Object> hp = map(settings.get("honeypot"));
        cfg.honeypotEnabled = bool(hp.get("enabled"), cfg.honeypotEnabled);
        cfg.minFormTimeSeconds = dbl(hp.get("min_form_time"), cfg.minFormTimeSeconds);

        Map<String, Object> ik = map(settings.get("ip_keyword_block"));
        cfg.ipKeywordBlockEnabled = bool(ik.get("enabled"), cfg.ipKeywordBlockEnabled);
        Set<String> mk = strSet(ik.get("malicious_keywords"));
        if (!mk.isEmpty()) cfg.blockedPathPatterns = new HashSet<>(mk);

        Map<String, Object> geo = map(settings.get("geo_block"));
        cfg.geoBlockEnabled = bool(geo.get("enabled"), cfg.geoBlockEnabled);
        cfg.geoBlockedCountries.addAll(upperSet(geo.get("block_countries")));
        cfg.geoAllowedCountries.addAll(upperSet(geo.get("allow_countries")));

        Map<String, Object> ai = map(settings.get("ai_anomaly"));
        cfg.aiEnabled = bool(ai.get("enabled"), cfg.aiEnabled);
        cfg.aiAnomalyScoreThreshold = dbl(ai.get("threshold"), cfg.aiAnomalyScoreThreshold);

        Map<String, Object> uuid = map(settings.get("uuid_tamper"));
        cfg.uuidTamperEnabled = bool(uuid.get("enabled"), cfg.uuidTamperEnabled);

        Map<String, Object> ex = map(settings.get("exemptions"));
        cfg.privateIpsExempted = bool(ex.get("private_ips_exempted"), cfg.privateIpsExempted);
        cfg.exemptIps.addAll(strSet(ex.get("auto_exempt_patterns")));
        cfg.legitimateRouteHints.addAll(strSet(settings.get("legitimate_route_hints")));
        cfg.legitimatePathKeywords.addAll(LegitimateRouteKeywordsCore.fromRouteHints(cfg.legitimateRouteHints));

        List<?> rules = list(settings.get("path_rules"));
        if (rules != null) {
            for (Object ruleObj : rules) {
                Map<String, Object> r = map(ruleObj);
                String prefix = str(or(r.get("PREFIX"), r.get("prefix")), null);
                if (prefix == null || prefix.isBlank()) continue;
                Set<String> disable = strSet(or(r.get("DISABLE"), r.get("disable")));

                Integer max = null;
                Integer window = null;
                Integer flood = null;
                Map<String, Map<String, Integer>> overrides = new HashMap<>();
                Map<String, Object> rate = map(or(r.get("RATE_LIMIT"), r.get("rate_limit")));
                if (!rate.isEmpty()) {
                    max = intObj(or(rate.get("MAX"), rate.get("max")));
                    window = intObj(or(rate.get("WINDOW"), rate.get("window")));
                    flood = intObj(or(rate.get("FLOOD"), rate.get("flood")));
                    Map<String, Integer> normalized = new HashMap<>();
                    if (max != null) normalized.put("max", max);
                    if (window != null) normalized.put("window", window);
                    if (flood != null) normalized.put("flood", flood);
                    overrides.put("rate_limit", normalized);
                }

                cfg.pathRules.add(new AiwafConfig.PathRule(
                        prefix, false, max, window, flood, disable, overrides
                ));
            }
        }
    }

    public static void applyEnv(AiwafConfig cfg, Map<String, String> env) {
        if (cfg == null || env == null) return;
        get(env, "AIWAF_RATE_WINDOW").ifPresent(v -> cfg.rateLimitWindowSeconds = parseInt(v, cfg.rateLimitWindowSeconds));
        get(env, "AIWAF_RATE_MAX").ifPresent(v -> cfg.rateLimitMax = parseInt(v, cfg.rateLimitMax));
        get(env, "AIWAF_RATE_FLOOD").ifPresent(v -> cfg.rateLimitFloodThreshold = parseInt(v, cfg.rateLimitFloodThreshold));
        get(env, "AIWAF_HEADER_VALIDATION").ifPresent(v -> cfg.headerValidationEnabled = parseBool(v, cfg.headerValidationEnabled));
        get(env, "AIWAF_HEADER_QUALITY_MIN_SCORE").ifPresent(v -> cfg.minHeaderQualityScore = parseInt(v, cfg.minHeaderQualityScore));
        get(env, "AIWAF_GEO_BLOCK_ENABLED").ifPresent(v -> cfg.geoBlockEnabled = parseBool(v, cfg.geoBlockEnabled));
        get(env, "AIWAF_GEO_BLOCK_COUNTRIES").ifPresent(v -> cfg.geoBlockedCountries.addAll(csvUpper(v)));
        get(env, "AIWAF_GEO_ALLOW_COUNTRIES").ifPresent(v -> cfg.geoAllowedCountries.addAll(csvUpper(v)));
        get(env, "AIWAF_AI_ENABLED").ifPresent(v -> cfg.aiEnabled = parseBool(v, cfg.aiEnabled));
        get(env, "AIWAF_AI_MODEL_PATH").ifPresent(v -> cfg.aiModelPath = v);
        get(env, "AIWAF_STORAGE_BACKEND").ifPresent(v -> cfg.storageBackend = v.toLowerCase(Locale.ROOT));
        get(env, "AIWAF_STORAGE_FILE_PATH").ifPresent(v -> cfg.storageFilePath = v);
    }

    private static Object or(Object a, Object b) { return a != null ? a : b; }
    private static Map<String, Object> map(Object o) {
        if (!(o instanceof Map<?, ?> m)) return Map.of();
        Map<String, Object> out = new HashMap<>();
        for (Map.Entry<?, ?> e : m.entrySet()) if (e.getKey() != null) out.put(String.valueOf(e.getKey()), e.getValue());
        return out;
    }
    private static List<?> list(Object o) { return (o instanceof List<?> l) ? l : null; }
    private static String str(Object o, String d) { return o == null ? d : String.valueOf(o); }
    private static boolean bool(Object o, boolean d) {
        if (o == null) return d;
        if (o instanceof Boolean b) return b;
        return parseBool(String.valueOf(o), d);
    }
    private static int intval(Object o, int d) {
        Integer v = intObj(o);
        return v == null ? d : v;
    }
    private static Integer intObj(Object o) {
        if (o == null) return null;
        if (o instanceof Number n) return n.intValue();
        try { return Integer.parseInt(String.valueOf(o)); } catch (Exception ex) { return null; }
    }
    private static double dbl(Object o, double d) {
        if (o == null) return d;
        if (o instanceof Number n) return n.doubleValue();
        try { return Double.parseDouble(String.valueOf(o)); } catch (Exception ex) { return d; }
    }
    private static Set<String> strSet(Object o) {
        Set<String> out = new HashSet<>();
        if (o instanceof List<?> l) for (Object v : l) if (v != null) out.add(String.valueOf(v));
        return out;
    }
    private static Set<String> upperSet(Object o) {
        Set<String> out = new HashSet<>();
        for (String s : strSet(o)) out.add(s.toUpperCase(Locale.ROOT));
        return out;
    }
    private static java.util.Optional<String> get(Map<String, String> env, String key) {
        for (Map.Entry<String, String> e : env.entrySet()) {
            if (key.equalsIgnoreCase(e.getKey())) return java.util.Optional.ofNullable(e.getValue());
        }
        return java.util.Optional.empty();
    }
    private static int parseInt(String v, int d) {
        try { return Integer.parseInt(v); } catch (Exception ex) { return d; }
    }
    private static boolean parseBool(String v, boolean d) {
        if (v == null) return d;
        String x = v.trim().toLowerCase(Locale.ROOT);
        if (Set.of("1", "true", "yes", "on").contains(x)) return true;
        if (Set.of("0", "false", "no", "off").contains(x)) return false;
        return d;
    }
    private static Set<String> csvUpper(String v) {
        Set<String> out = new HashSet<>();
        if (v == null || v.isBlank()) return out;
        for (String p : v.split(",")) {
            String t = p.trim();
            if (!t.isBlank()) out.add(t.toUpperCase(Locale.ROOT));
        }
        return out;
    }
}
