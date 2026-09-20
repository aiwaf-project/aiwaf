package com.aiwaf.core;

import com.maxmind.db.Reader;

import java.io.InputStream;
import java.net.InetAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import java.util.function.Function;

public final class GeoIpCore {
    private static final String RESOURCE_PATH = "geolock/ipinfo_lite.mmdb";
    private static final String CLASSPATH_PATH = "classpath:" + RESOURCE_PATH;
    private static final Map<String, Reader> READERS = new ConcurrentHashMap<>();
    private static final Map<String, CachedCountry> COUNTRY_CACHE = new ConcurrentHashMap<>();

    private GeoIpCore() {}

    public static String defaultMmdbPath() {
        try {
            var url = GeoIpCore.class.getClassLoader().getResource(RESOURCE_PATH);
            if (url != null && "file".equalsIgnoreCase(url.getProtocol())) {
                return Path.of(url.toURI()).toString();
            }
        } catch (Exception ignored) {
        }
        return CLASSPATH_PATH;
    }

    public static String lookupCountry(String ip, String dbPath) {
        Object raw = lookupRaw(ip, dbPath);
        String country = extractCountryFromRaw(raw);
        if (country == null && raw instanceof Map<?, ?> map) {
            country = extractCountryFromRaw(map.get("registered_country"));
        }
        return country == null ? null : country.trim().toUpperCase(Locale.ROOT);
    }

    public static String lookupCountryName(String ip, String dbPath) {
        Object raw = lookupRaw(ip, dbPath);
        String country = extractCountryNameFromRaw(raw);
        if (country == null && raw instanceof Map<?, ?> map) {
            country = extractCountryNameFromRaw(map.get("registered_country"));
        }
        return country == null ? null : country.trim();
    }

    public static boolean isMmdbLookupAvailable() {
        return GeoIpCore.class.getClassLoader().getResource(RESOURCE_PATH) != null;
    }

    /** Fast runtime lookup with a bounded, expiring in-process cache. */
    public static String lookupCountryCached(String ip, String dbPath, int cacheSeconds, int maxEntries) {
        if (ip == null || ip.isBlank()) return null;
        String source = sourceKey(dbPath);
        String key = source + "|" + ip.trim();
        long now = System.currentTimeMillis();
        CachedCountry cached = COUNTRY_CACHE.get(key);
        if (cached != null && cached.expiresAtMillis() >= now) return cached.value();
        if (cached != null) COUNTRY_CACHE.remove(key, cached);
        String value = lookupCountry(ip, dbPath);
        if (maxEntries > 0 && COUNTRY_CACHE.size() >= maxEntries) COUNTRY_CACHE.clear();
        long ttlMillis = Math.max(1, cacheSeconds) * 1000L;
        COUNTRY_CACHE.put(key, new CachedCountry(value, now + ttlMillis));
        return value;
    }

    public static String lookupCountryCached(
            String ip,
            String dbPath,
            String cacheKey,
            int cacheSeconds,
            Function<String, String> cacheGet,
            Consumer<CacheSetCall> cacheSet
    ) {
        if (cacheGet != null && cacheKey != null) {
            String cached = cacheGet.apply(cacheKey);
            if (cached != null) {
                return cached;
            }
        }
        String value = lookupCountry(ip, dbPath);
        // Keep parity with Python core: cache_set is called even when lookup returns null.
        if (cacheSet != null && cacheKey != null) {
            cacheSet.accept(new CacheSetCall(cacheKey, value, cacheSeconds));
        }
        return value;
    }

    static String extractCountryFromRaw(Object raw) {
        if (!(raw instanceof Map<?, ?> map)) {
            return null;
        }
        Object code = firstNonEmpty(
                map.get("iso_code"),
                map.get("country_code"),
                map.get("country_code2"),
                map.get("country_code3")
        );
        if (code != null) {
            return String.valueOf(code);
        }
        Object country = map.get("country");
        if (country instanceof Map<?, ?> countryMap) {
            Object iso = countryMap.get("iso_code");
            if (iso != null && !String.valueOf(iso).isBlank()) {
                return String.valueOf(iso);
            }
        }
        if (country instanceof String s && s.length() >= 2) {
            return s;
        }
        return null;
    }

    static String extractCountryNameFromRaw(Object raw) {
        if (!(raw instanceof Map<?, ?> map)) {
            return null;
        }
        Object country = map.get("country");
        if (country instanceof Map<?, ?> countryMap) {
            Object name = countryMap.get("name");
            if (name != null && !String.valueOf(name).isBlank()) {
                return String.valueOf(name);
            }
        }
        if (country instanceof String s && s.length() >= 2) {
            return s;
        }
        Object names = map.get("names");
        if (names instanceof Map<?, ?> namesMap) {
            Object english = namesMap.get("en");
            if (english != null && !String.valueOf(english).isBlank()) return String.valueOf(english);
        }
        Object countryName = map.get("country_name");
        if (countryName != null && !String.valueOf(countryName).isBlank()) {
            return String.valueOf(countryName);
        }
        return null;
    }

    private static Object lookupRaw(String ip, String dbPath) {
        if (ip == null || ip.isBlank()) return null;
        try {
            Reader reader = reader(dbPath);
            return reader == null ? null : reader.get(InetAddress.getByName(ip.trim()), Map.class);
        } catch (Exception ignored) {
            return null;
        }
    }

    private static Reader reader(String dbPath) {
        String key = sourceKey(dbPath);
        Reader existing = READERS.get(key);
        if (existing != null) return existing;
        synchronized (READERS) {
            existing = READERS.get(key);
            if (existing != null) return existing;
            try {
                Reader opened;
                if (CLASSPATH_PATH.equals(key)) {
                    InputStream stream = GeoIpCore.class.getClassLoader().getResourceAsStream(RESOURCE_PATH);
                    if (stream == null) return null;
                    try (stream) {
                        opened = new Reader(stream);
                    }
                } else {
                    Path path = Path.of(key);
                    if (!Files.isRegularFile(path)) return null;
                    opened = new Reader(path.toFile());
                }
                READERS.put(key, opened);
                return opened;
            } catch (Exception ignored) {
                return null;
            }
        }
    }

    private static String sourceKey(String dbPath) {
        if (dbPath == null || dbPath.isBlank() || dbPath.equals(RESOURCE_PATH)
                || dbPath.equals(CLASSPATH_PATH)) return CLASSPATH_PATH;
        try {
            return Path.of(dbPath).toAbsolutePath().normalize().toString();
        } catch (RuntimeException ignored) {
            return dbPath;
        }
    }

    private static Object firstNonEmpty(Object... values) {
        if (values == null) {
            return null;
        }
        for (Object value : values) {
            if (value != null && !String.valueOf(value).isBlank()) {
                return value;
            }
        }
        return null;
    }

    public record CacheSetCall(String key, String value, int timeoutSeconds) {}
    private record CachedCountry(String value, long expiresAtMillis) {}
}
