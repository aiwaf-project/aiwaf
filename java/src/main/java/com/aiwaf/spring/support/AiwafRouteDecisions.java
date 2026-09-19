package com.aiwaf.spring.support;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.ExemptionsCore;
import com.aiwaf.spring.annotations.AiwafExempt;
import com.aiwaf.spring.annotations.AiwafExemptFrom;
import com.aiwaf.spring.annotations.AiwafOnly;
import com.aiwaf.spring.annotations.AiwafRequireProtection;
import org.springframework.web.method.HandlerMethod;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class AiwafRouteDecisions {
    private AiwafRouteDecisions() {}

    public static boolean shouldApply(String path, Object handler, String middlewareName, java.util.List<AiwafConfig.PathRule> pathRules) {
        String mw = ExemptionsCore.normalizeMiddlewareName(middlewareName);

        if (!(handler instanceof HandlerMethod hm)) {
            AiwafConfig.PathRule rule = ExemptionsCore.getPathRuleForPath(path, pathRules);
            if (rule != null && rule.disables(mw)) {
                return false;
            }
            return true;
        }

        Set<String> required = normalize(values(hm, AiwafRequireProtection.class));
        if (required.contains(mw)) return true;

        AiwafConfig.PathRule rule = ExemptionsCore.getPathRuleForPath(path, pathRules);
        if (rule != null && rule.disables(mw)) {
            return false;
        }

        if (has(hm, AiwafExempt.class)) return false;

        Set<String> exemptFrom = normalize(values(hm, AiwafExemptFrom.class));
        if (exemptFrom.contains(mw)) return false;

        Set<String> only = normalize(values(hm, AiwafOnly.class));
        if (!only.isEmpty()) {
            return only.contains(mw);
        }

        return true;
    }

    private static <A extends java.lang.annotation.Annotation> boolean has(HandlerMethod hm, Class<A> ann) {
        return hm.getMethod().isAnnotationPresent(ann)
                || hm.getBeanType().isAnnotationPresent(ann);
    }

    private static <A extends java.lang.annotation.Annotation> String[] values(HandlerMethod hm, Class<A> ann) {
        if (ann == AiwafExemptFrom.class) {
            List<String> merged = new ArrayList<>();
            AiwafExemptFrom c = hm.getBeanType().getAnnotation(AiwafExemptFrom.class);
            if (c != null) {
                merged.addAll(List.of(c.value()));
            }
            AiwafExemptFrom m = hm.getMethodAnnotation(AiwafExemptFrom.class);
            if (m != null) {
                merged.addAll(List.of(m.value()));
            }
            return merged.toArray(new String[0]);
        }
        if (ann == AiwafOnly.class) {
            List<String> merged = new ArrayList<>();
            AiwafOnly c = hm.getBeanType().getAnnotation(AiwafOnly.class);
            if (c != null) {
                merged.addAll(List.of(c.value()));
            }
            AiwafOnly m = hm.getMethodAnnotation(AiwafOnly.class);
            if (m != null) {
                merged.addAll(List.of(m.value()));
            }
            return merged.toArray(new String[0]);
        }
        if (ann == AiwafRequireProtection.class) {
            List<String> merged = new ArrayList<>();
            AiwafRequireProtection c = hm.getBeanType().getAnnotation(AiwafRequireProtection.class);
            if (c != null) {
                merged.addAll(List.of(c.value()));
            }
            AiwafRequireProtection m = hm.getMethodAnnotation(AiwafRequireProtection.class);
            if (m != null) {
                merged.addAll(List.of(m.value()));
            }
            return merged.toArray(new String[0]);
        }
        return new String[]{};
    }

    private static Set<String> normalize(String[] values) {
        Set<String> out = new HashSet<>();
        if (values == null) return out;
        for (String v : values) {
            out.add(ExemptionsCore.normalizeMiddlewareName(v));
        }
        return out;
    }
}
