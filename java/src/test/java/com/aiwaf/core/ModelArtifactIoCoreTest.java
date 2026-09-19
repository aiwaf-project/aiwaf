package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelArtifactIoCoreTest {

    @Test
    void save_and_load_compatible_model() throws Exception {
        List<NormalizedEvent> events = List.of(
                new NormalizedEvent("1.1.1.1", "GET", "/home", 200, 3, LocalDateTime.now(), "", "", "", true, false),
                new NormalizedEvent("1.1.1.2", "GET", "/about", 200, 3, LocalDateTime.now(), "", "", "", true, false),
                new NormalizedEvent("1.1.1.3", "GET", "/pricing", 200, 3, LocalDateTime.now(), "", "", "", true, false)
        );
        TrainedModelCore model = TrainingCore.trainModel(events, List.of(".php"));
        String file = Files.createTempFile("aiwaf-model-compat-", ".bin").toString();
        assertTrue(ModelArtifactIoCore.save(model, file));
        assertNotNull(ModelArtifactIoCore.load(file));
    }

    @Test
    void legacy_nine_feature_java_model_remains_readable() throws Exception {
        double[][] rows = new double[][]{new double[9], new double[9], new double[9]};
        rows[1][0] = 1;
        rows[2][0] = 2;
        IsolationForestCore.Model forest = IsolationForestCore.fit(rows, 8, 3, 42L);
        Map<String, Object> iforest = new java.util.HashMap<>();
        iforest.put("model", forest);
        iforest.put("feature_names", List.of(
                "burst_count", "kw_hits", "method_is_post", "path_depth", "path_len",
                "response_time_ms", "status_code", "status_idx", "total_404"
        ));
        Map<String, Object> payload = new java.util.HashMap<>();
        payload.put("isolation_forest", iforest);
        payload.put("metadata", Map.of("model_schema", "iforest-v1"));
        String file = Files.createTempFile("aiwaf-legacy-nine-", ".bin").toString();
        assertTrue(ModelArtifactIoCore.save(new TrainedModelCore("isolation-forest", "1", payload), file));
        assertNotNull(ModelArtifactIoCore.load(file));
    }

    @Test
    void incompatible_model_type_is_rejected() throws Exception {
        TrainedModelCore wrong = new TrainedModelCore("baseline-statistical", "1", java.util.Map.of());
        String file = Files.createTempFile("aiwaf-model-incompat-", ".bin").toString();
        assertTrue(ModelArtifactIoCore.save(wrong, file));
        assertNull(ModelArtifactIoCore.load(file));
    }

    @Test
    void old_iforest_schema_is_migrated_to_v1() throws Exception {
        IsolationForestCore.Model tiny = IsolationForestCore.fit(new double[][]{
                {0.0, 0.0},
                {1.0, 1.0},
                {2.0, 2.0}
        }, 8, 3, 42L);

        Map<String, Object> oldIf = new java.util.HashMap<>();
        oldIf.put("model", tiny);
        oldIf.put("sampleSize", tiny.sampleSize());
        oldIf.put("featureNames", List.of("path_len", "kw_hits"));
        oldIf.put("anomalyCount", 1);
        oldIf.put("modelBackend", "aiwaf_java");

        Map<String, Object> oldMeta = new java.util.HashMap<>();
        oldMeta.put("model_schema", "iforest-v0");

        Map<String, Object> payload = new java.util.HashMap<>();
        payload.put("isolation_forest", oldIf);
        payload.put("metadata", oldMeta);
        TrainedModelCore old = new TrainedModelCore("isolation-forest", "1", payload);

        String file = Files.createTempFile("aiwaf-model-migrate-", ".bin").toString();
        assertTrue(ModelArtifactIoCore.save(old, file));

        TrainedModelCore loaded = ModelArtifactIoCore.load(file);
        assertNotNull(loaded);
        @SuppressWarnings("unchecked")
        Map<String, Object> loadedIf = (Map<String, Object>) loaded.payload().get("isolation_forest");
        @SuppressWarnings("unchecked")
        Map<String, Object> loadedMeta = (Map<String, Object>) loaded.payload().get("metadata");
        assertEquals(3, ((Number) loadedIf.get("sample_size")).intValue());
        assertEquals("iforest-v1", loadedMeta.get("model_schema"));
    }
}
