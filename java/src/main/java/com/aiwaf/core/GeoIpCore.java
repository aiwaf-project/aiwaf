package com.aiwaf.core;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class GeoIpCore {
    private static volatile Boolean mmdbLookupAvailable;

    private GeoIpCore() {}

    public static String defaultMmdbPath() {
        try {
            var url = GeoIpCore.class.getClassLoader().getResource("geolock/ipinfo_lite.mmdb");
            if (url != null) {
                return Path.of(url.toURI()).toString();
            }
        } catch (Exception ignored) {
        }
        return "geolock/ipinfo_lite.mmdb";
    }

    public static String lookupCountry(String ip, String dbPath) {
        if (dbPath == null || dbPath.isBlank() || ip == null || ip.isBlank()) {
            return null;
        }
        if (!isMmdbLookupAvailable()) {
            return null;
        }
        Path p = Path.of(dbPath);
        if (!Files.exists(p)) {
            return null;
        }
        return lookupCountryViaMmdbLookup(ip, dbPath);
    }

    public static String lookupCountryName(String ip, String dbPath) {
        if (dbPath == null || dbPath.isBlank() || ip == null || ip.isBlank()) {
            return null;
        }
        if (!isMmdbLookupAvailable()) {
            return null;
        }
        Path p = Path.of(dbPath);
        if (!Files.exists(p)) {
            return null;
        }
        return lookupCountryNameViaMmdbLookup(ip, dbPath);
    }

    public static boolean isMmdbLookupAvailable() {
        Boolean cached = mmdbLookupAvailable;
        if (cached != null) {
            return cached;
        }
        synchronized (GeoIpCore.class) {
            if (mmdbLookupAvailable != null) {
                return mmdbLookupAvailable;
            }
            mmdbLookupAvailable = probeMmdbLookup();
            return mmdbLookupAvailable;
        }
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
        Object countryName = map.get("country_name");
        if (countryName != null && !String.valueOf(countryName).isBlank()) {
            return String.valueOf(countryName);
        }
        return null;
    }

    private static String lookupCountryViaMmdbLookup(String ip, String dbPath) {
        String code = runMmdbLookup(dbPath, ip, "country", "iso_code");
        if (code == null) {
            code = runMmdbLookup(dbPath, ip, "registered_country", "iso_code");
        }
        if (code == null) {
            code = runMmdbLookup(dbPath, ip, "country_code");
        }
        if (code == null) {
            code = runMmdbLookup(dbPath, ip, "country_code2");
        }
        if (code == null) {
            code = runMmdbLookup(dbPath, ip, "country_code3");
        }
        return code == null ? null : code.trim().toUpperCase(Locale.ROOT);
    }

    private static String lookupCountryNameViaMmdbLookup(String ip, String dbPath) {
        String name = runMmdbLookup(dbPath, ip, "country", "names", "en");
        if (name == null) {
            name = runMmdbLookup(dbPath, ip, "registered_country", "names", "en");
        }
        if (name == null) {
            name = runMmdbLookup(dbPath, ip, "country_name");
        }
        return name == null ? null : name.trim();
    }

    private static String runMmdbLookup(String dbPath, String ip, String... pathParts) {
        try {
            java.util.List<String> cmd = new java.util.ArrayList<>();
            cmd.add("mmdblookup");
            cmd.add("--file");
            cmd.add(dbPath);
            cmd.add("--ip");
            cmd.add(ip);
            if (pathParts != null) {
                for (String pathPart : pathParts) {
                    if (pathPart != null && !pathPart.isBlank()) {
                        cmd.add(pathPart);
                    }
                }
            }

            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            Process proc = pb.start();
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append('\n');
                }
            }
            boolean finished = proc.waitFor(3, TimeUnit.SECONDS);
            if (!finished) {
                proc.destroyForcibly();
                return null;
            }
            return extractQuotedValue(output.toString());
        } catch (Exception ignored) {
            return null;
        }
    }

    private static String extractQuotedValue(String text) {
        if (text == null || text.isBlank()) {
            return null;
        }
        Matcher m = Pattern.compile("\"([^\"]+)\"").matcher(text);
        if (m.find()) {
            String value = m.group(1);
            if (!value.isBlank()) {
                return value;
            }
        }
        return null;
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

    private static boolean probeMmdbLookup() {
        try {
            ProcessBuilder pb = new ProcessBuilder("mmdblookup", "--version");
            pb.redirectErrorStream(true);
            Process proc = pb.start();
            boolean finished = proc.waitFor(2, TimeUnit.SECONDS);
            if (!finished) {
                proc.destroyForcibly();
                return false;
            }
            int rc = proc.exitValue();
            return rc == 0;
        } catch (Exception ignored) {
            return false;
        }
    }

    public record CacheSetCall(String key, String value, int timeoutSeconds) {}
}
