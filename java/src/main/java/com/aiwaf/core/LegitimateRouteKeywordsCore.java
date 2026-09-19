package com.aiwaf.core;

import java.lang.reflect.Method;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

public final class LegitimateRouteKeywordsCore {
    private LegitimateRouteKeywordsCore() {}

    public static Set<String> fromRouteHints(Iterable<String> hints) {
        Set<String> out = new HashSet<>();
        if (hints == null) return out;
        for (String hint : hints) {
            addTokenized(out, hint);
        }
        return out;
    }

    public static Set<String> fromHandlerClasses(Iterable<Class<?>> handlerTypes) {
        Set<String> out = new HashSet<>();
        if (handlerTypes == null) return out;
        for (Class<?> type : handlerTypes) {
            if (type == null) continue;
            addTokenized(out, type.getSimpleName());
            addTokenized(out, type.getName());
            for (Method m : type.getMethods()) {
                addTokenized(out, m.getName());
            }
        }
        return out;
    }

    public static void mergeInto(Set<String> target, Set<String> additional) {
        if (target == null || additional == null || additional.isEmpty()) return;
        target.addAll(additional);
    }

    static void addTokenized(Set<String> out, String text) {
        if (text == null || text.isBlank()) return;
        String[] parts = splitCamelAndSymbols(text);
        for (String part : parts) {
            String p = part.toLowerCase(Locale.ROOT).trim();
            if (p.length() < 3) continue;
            out.add(p);
            if (!p.endsWith("s")) out.add(p + "s");
        }
    }

    static String[] splitCamelAndSymbols(String text) {
        String deCamel = text
                .replaceAll("([a-z0-9])([A-Z])", "$1 $2")
                .replaceAll("([A-Z])([A-Z][a-z])", "$1 $2");
        return deCamel.split("[^A-Za-z0-9]+");
    }
}
