package com.aiwaf.core;

import java.util.UUID;
import java.lang.annotation.Annotation;
import java.lang.reflect.AnnotatedElement;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public final class UuidPolicyCore {
    private UuidPolicyCore() {}

    public static UUIDTamperDecision evaluateUuidTamper(String uuidValue, UUIDLookup lookup) {
        if (uuidValue == null || uuidValue.isBlank()) {
            return new UUIDTamperDecision(true, null);
        }
        try {
            UUID.fromString(uuidValue);
        } catch (IllegalArgumentException ex) {
            return new UUIDTamperDecision(false, "invalid_uuid_format");
        }
        if (lookup == null) {
            return new UUIDTamperDecision(true, null);
        }
        if (lookup.exists(uuidValue)) {
            return new UUIDTamperDecision(true, null);
        }
        return new UUIDTamperDecision(false, "uuid_not_found");
    }

    public interface UUIDLookup {
        boolean exists(String uuidValue);
    }

    /** A UUID primary key or unique UUID property discovered from persistence annotations. */
    public record UUIDModelField(Class<?> modelClass, String fieldName, boolean primaryKey) {}

    @FunctionalInterface
    public interface UUIDModelFieldLookup {
        boolean exists(Class<?> modelClass, String fieldName, UUID value);
    }

    public static List<UUIDModelField> collectUuidModelFields(Collection<Class<?>> modelClasses) {
        List<UUIDModelField> out = new ArrayList<>();
        if (modelClasses == null) return out;
        for (Class<?> modelClass : modelClasses) {
            if (modelClass == null) continue;
            for (Field field : modelClass.getDeclaredFields()) {
                if (!UUID.class.equals(field.getType())) continue;
                boolean primary = hasAnnotation(field, "jakarta.persistence.Id", "javax.persistence.Id");
                if (primary || isUniqueColumn(field)) {
                    out.add(new UUIDModelField(modelClass, field.getName(), primary));
                }
            }
            for (Method method : modelClass.getDeclaredMethods()) {
                if (method.getParameterCount() != 0 || !UUID.class.equals(method.getReturnType())) continue;
                boolean primary = hasAnnotation(method, "jakarta.persistence.Id", "javax.persistence.Id");
                if (primary || isUniqueColumn(method)) {
                    String name = propertyName(method.getName());
                    if (out.stream().noneMatch(item -> item.modelClass().equals(modelClass) && item.fieldName().equals(name))) {
                        out.add(new UUIDModelField(modelClass, name, primary));
                    }
                }
            }
        }
        return List.copyOf(out);
    }

    public static boolean uuidExistsInModelFields(
            String value,
            Collection<UUIDModelField> fields,
            UUIDModelFieldLookup lookup
    ) {
        if (lookup == null || fields == null) return false;
        final UUID uuid;
        try {
            uuid = UUID.fromString(value);
        } catch (Exception ex) {
            return false;
        }
        for (UUIDModelField field : fields) {
            try {
                if (lookup.exists(field.modelClass(), field.fieldName(), uuid)) return true;
            } catch (IllegalArgumentException ignored) {
                // Match Python's behavior: one incompatible field must not abort the remaining lookups.
            }
        }
        return false;
    }

    private static boolean hasAnnotation(AnnotatedElement element, String... names) {
        for (Annotation annotation : element.getAnnotations()) {
            for (String name : names) {
                if (name.equals(annotation.annotationType().getName())) return true;
            }
        }
        return false;
    }

    private static boolean isUniqueColumn(AnnotatedElement element) {
        for (Annotation annotation : element.getAnnotations()) {
            String name = annotation.annotationType().getName();
            if (!"jakarta.persistence.Column".equals(name) && !"javax.persistence.Column".equals(name)) continue;
            try {
                return Boolean.TRUE.equals(annotation.annotationType().getMethod("unique").invoke(annotation));
            } catch (ReflectiveOperationException ignored) {
                return false;
            }
        }
        return false;
    }

    private static String propertyName(String methodName) {
        String raw = methodName.startsWith("get") && methodName.length() > 3
                ? methodName.substring(3)
                : methodName;
        if (raw.isEmpty()) return raw;
        return Character.toLowerCase(raw.charAt(0)) + raw.substring(1);
    }

    public record UUIDTamperDecision(boolean allow, String reason) {}
}
