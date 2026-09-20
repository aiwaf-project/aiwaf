package com.aiwaf.core;

import java.util.HashMap;
import java.util.Map;

public final class ModelArtifactMigrationCore {
    private ModelArtifactMigrationCore() {}

    public static TrainedModelCore migrate(TrainedModelCore model) {
        if (model == null || model.payload() == null) {
            return model;
        }
        if (!"isolation-forest".equals(model.modelType())) {
            return model;
        }
        Object ifObj = model.payload().get("isolation_forest");
        if (!(ifObj instanceof Map<?, ?> rawIf)) {
            return model;
        }

        Map<String, Object> payload = new HashMap<>(model.payload());
        Map<String, Object> iforest = toStringObjectMap(rawIf);
        iforest = migrateIsolationForestKeys(iforest);
        payload.put("isolation_forest", iforest);

        Map<String, Object> metadata = metadataWithDefaults(payload);
        payload.put("metadata", metadata);
        return new TrainedModelCore(model.modelType(), model.version(), payload);
    }

    private static Map<String, Object> metadataWithDefaults(Map<String, Object> payload) {
        Object metadataObj = payload.get("metadata");
        Map<String, Object> metadata = metadataObj instanceof Map<?, ?> m ? toStringObjectMap(m) : new HashMap<>();
        String schema = String.valueOf(metadata.getOrDefault("model_schema", "iforest-v1"));
        if ("iforest-v0".equals(schema)) {
            schema = "iforest-v1";
        }
        metadata.put("model_schema", schema);
        metadata.putIfAbsent("model_backend", "aiwaf_java");
        metadata.putIfAbsent("aiwaf_java_version", "1.2.0");
        metadata.putIfAbsent("created_at_epoch_ms", System.currentTimeMillis());
        metadata.putIfAbsent("java_runtime_version", System.getProperty("java.runtime.version", "unknown"));
        return metadata;
    }

    private static Map<String, Object> migrateIsolationForestKeys(Map<String, Object> input) {
        Map<String, Object> out = new HashMap<>(input);
        rename(out, "sampleSize", "sample_size");
        rename(out, "featureNames", "feature_names");
        rename(out, "anomalyCount", "anomaly_count");
        rename(out, "modelBackend", "backend");
        return out;
    }

    private static void rename(Map<String, Object> map, String oldKey, String newKey) {
        if (!map.containsKey(newKey) && map.containsKey(oldKey)) {
            map.put(newKey, map.get(oldKey));
        }
    }

    private static Map<String, Object> toStringObjectMap(Map<?, ?> in) {
        Map<String, Object> out = new HashMap<>();
        for (Map.Entry<?, ?> e : in.entrySet()) {
            if (e.getKey() != null) {
                out.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        return out;
    }
}
