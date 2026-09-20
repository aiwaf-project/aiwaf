package com.aiwaf.spring;

import com.aiwaf.core.UuidPolicyCore;
import org.springframework.context.ApplicationContext;

import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** Optional reflection-based JPA UUID lookup without forcing a JPA dependency on AIWAF. */
public final class SpringUuidModelLookup implements UuidPolicyCore.UUIDLookup {
    private record Binding(Object entityManagerFactory, UuidPolicyCore.UUIDModelField field) {}

    private final List<Binding> bindings;

    private SpringUuidModelLookup(List<Binding> bindings) {
        this.bindings = List.copyOf(bindings);
    }

    /**
     * Discover UUID primary keys and unique UUID properties from every JPA metamodel in a Spring
     * context. Returns {@code null} when JPA or qualifying fields are absent.
     */
    @SuppressWarnings({"rawtypes", "unchecked"})
    public static SpringUuidModelLookup discover(ApplicationContext context) {
        if (context == null) return null;
        Class<?> factoryType = loadClass("jakarta.persistence.EntityManagerFactory");
        if (factoryType == null) factoryType = loadClass("javax.persistence.EntityManagerFactory");
        if (factoryType == null) return null;

        List<Binding> bindings = new ArrayList<>();
        Map<String, ?> factories = context.getBeansOfType((Class) factoryType);
        for (Object factory : factories.values()) {
            for (Class<?> entityClass : entityClasses(factory)) {
                for (UuidPolicyCore.UUIDModelField field :
                        UuidPolicyCore.collectUuidModelFields(List.of(entityClass))) {
                    bindings.add(new Binding(factory, field));
                }
            }
        }
        return bindings.isEmpty() ? null : new SpringUuidModelLookup(bindings);
    }

    public List<UuidPolicyCore.UUIDModelField> fields() {
        return bindings.stream().map(Binding::field).toList();
    }

    @Override
    public boolean exists(String uuidValue) {
        final UUID uuid;
        try {
            uuid = UUID.fromString(uuidValue);
        } catch (Exception ex) {
            return false;
        }
        boolean queryCompleted = false;
        RuntimeException lastFailure = null;
        for (Binding binding : bindings) {
            Object entityManager = null;
            try {
                entityManager = invoke(binding.entityManagerFactory(), "createEntityManager");
                UuidPolicyCore.UUIDModelField field = binding.field();
                if (field.primaryKey()) {
                    Object found = entityManager.getClass()
                            .getMethod("find", Class.class, Object.class)
                            .invoke(entityManager, field.modelClass(), uuid);
                    queryCompleted = true;
                    if (found != null) return true;
                    continue;
                }
                String queryText = "select count(e) from " + entityName(field.modelClass())
                        + " e where e." + field.fieldName() + " = :uuid";
                Object query = entityManager.getClass().getMethod("createQuery", String.class)
                        .invoke(entityManager, queryText);
                query = query.getClass().getMethod("setParameter", String.class, Object.class)
                        .invoke(query, "uuid", uuid);
                Object result = query.getClass().getMethod("getSingleResult").invoke(query);
                queryCompleted = true;
                if (result instanceof Number number && number.longValue() > 0) return true;
            } catch (ReflectiveOperationException | RuntimeException failure) {
                // Try the next entity/field, matching Python's tolerant multi-model lookup.
                lastFailure = new IllegalStateException("Unable to query UUID model field", failure);
            } finally {
                if (entityManager != null) {
                    try { invoke(entityManager, "close"); } catch (Exception ignored) {}
                }
            }
        }
        if (!queryCompleted && lastFailure != null) throw lastFailure;
        return false;
    }

    private static Collection<Class<?>> entityClasses(Object factory) {
        List<Class<?>> out = new ArrayList<>();
        try {
            Object metamodel = invoke(factory, "getMetamodel");
            Object entities = invoke(metamodel, "getEntities");
            if (entities instanceof Iterable<?> values) {
                for (Object entity : values) {
                    Object javaType = invoke(entity, "getJavaType");
                    if (javaType instanceof Class<?> cls) out.add(cls);
                }
            }
        } catch (ReflectiveOperationException ignored) {
        }
        return out;
    }

    private static String entityName(Class<?> modelClass) {
        for (Annotation annotation : modelClass.getAnnotations()) {
            String type = annotation.annotationType().getName();
            if (!"jakarta.persistence.Entity".equals(type) && !"javax.persistence.Entity".equals(type)) continue;
            try {
                Object configured = annotation.annotationType().getMethod("name").invoke(annotation);
                if (configured != null && !String.valueOf(configured).isBlank()) return String.valueOf(configured);
            } catch (ReflectiveOperationException ignored) {
            }
        }
        return modelClass.getSimpleName();
    }

    private static Object invoke(Object target, String method) throws ReflectiveOperationException {
        Method m = target.getClass().getMethod(method);
        return m.invoke(target);
    }

    private static Class<?> loadClass(String name) {
        try { return Class.forName(name); } catch (ClassNotFoundException ignored) { return null; }
    }
}
