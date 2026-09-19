package com.aiwaf.core;

import java.util.Collection;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

public final class GeoCore {
    private GeoCore() {}

    public static boolean shouldGeoBlock(
            String countryCode,
            Collection<String> allowCountries,
            Collection<String> blockCountries
    ) {
        if (countryCode == null || countryCode.isBlank()) {
            return false;
        }
        String code = countryCode.toUpperCase(Locale.ROOT);
        Set<String> allow = normalize(allowCountries);
        Set<String> block = normalize(blockCountries);
        if (!allow.isEmpty()) {
            return !allow.contains(code);
        }
        return block.contains(code);
    }

    private static Set<String> normalize(Collection<String> values) {
        Set<String> out = new HashSet<>();
        if (values == null) {
            return out;
        }
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                out.add(value.toUpperCase(Locale.ROOT));
            }
        }
        return out;
    }
}
