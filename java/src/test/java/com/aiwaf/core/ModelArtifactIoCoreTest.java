package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
        String json = Files.readString(java.nio.file.Path.of(file), StandardCharsets.UTF_8);
        assertTrue(json.contains("\"model_type\":\"aiwaf_rust.IsolationForest\""));
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
        assertTrue(ModelArtifactIoCore.saveLegacy(new TrainedModelCore("isolation-forest", "1", payload), file));
        assertNotNull(ModelArtifactIoCore.load(file));
    }

    @Test
    void incompatible_model_type_is_rejected() throws Exception {
        TrainedModelCore wrong = new TrainedModelCore("baseline-statistical", "1", java.util.Map.of());
        String file = Files.createTempFile("aiwaf-model-incompat-", ".bin").toString();
        assertFalse(ModelArtifactIoCore.save(wrong, file));
        assertEquals(0L, Files.size(java.nio.file.Path.of(file)));
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
        assertTrue(ModelArtifactIoCore.saveLegacy(old, file));

        TrainedModelCore loaded = ModelArtifactIoCore.load(file);
        assertNotNull(loaded);
        @SuppressWarnings("unchecked")
        Map<String, Object> loadedIf = (Map<String, Object>) loaded.payload().get("isolation_forest");
        @SuppressWarnings("unchecked")
        Map<String, Object> loadedMeta = (Map<String, Object>) loaded.payload().get("metadata");
        assertEquals(3, ((Number) loadedIf.get("sample_size")).intValue());
        assertEquals("iforest-v1", loadedMeta.get("model_schema"));
    }

    @Test
    void loads_python_rust_json_artifact() throws Exception {
        String json = """
                {
                  "model_type":"aiwaf_rust.IsolationForest",
                  "model_backend":"aiwaf_rust",
                  "feature_count":6,
                  "samples_count":2,
                  "framework":"flask",
                  "backend":"rust",
                  "model_state":{
                    "nEstimators":1,
                    "maxSamples":2,
                    "contamination":"auto",
                    "maxFeatures":6,
                    "bootstrap":false,
                    "randomState":42,
                    "verbose":0,
                    "warmStart":false,
                    "maxSamples_":2,
                    "maxFeatures_":6,
                    "offset_":-0.5,
                    "nFeaturesIn_":6,
                    "trees":[{
                      "depth":0,"maxDepth":1,"splitAttr":0,"splitValue":10.0,"size":2,
                      "left":{"depth":1,"maxDepth":1,"splitAttr":null,"splitValue":null,"size":1,"left":null,"right":null},
                      "right":{"depth":1,"maxDepth":1,"splitAttr":null,"splitValue":null,"size":1,"left":null,"right":null}
                    }],
                    "estimatorsFeatures":[[0,1,2,3,4,5]]
                  }
                }
                """;
        java.nio.file.Path file = Files.createTempFile("aiwaf-python-rust-", ".json");
        Files.writeString(file, json, StandardCharsets.UTF_8);

        TrainedModelCore loaded = ModelArtifactIoCore.load(file.toString());
        assertNotNull(loaded);
        @SuppressWarnings("unchecked")
        Map<String, Object> forest = (Map<String, Object>) loaded.payload().get("isolation_forest");
        IsolationForestCore.Model model = (IsolationForestCore.Model) forest.get("model");
        assertEquals(0.5, IsolationForestCore.score(model, new double[]{5, 0, 0, 0, 0, 0}), 1e-12);
        assertEquals(FeaturesCore.PYTHON_FEATURE_NAMES, forest.get("feature_names"));
    }

    @Test
    void portable_round_trip_preserves_scores() throws Exception {
        List<NormalizedEvent> events = List.of(
                new NormalizedEvent("1.1.1.1", "GET", "/home", 200, 3, LocalDateTime.now(), "", "", "", true, false),
                new NormalizedEvent("1.1.1.2", "GET", "/about", 200, 4, LocalDateTime.now(), "", "", "", true, false),
                new NormalizedEvent("1.1.1.3", "GET", "/pricing", 404, 5, LocalDateTime.now(), "", "", "", false, false)
        );
        TrainedModelCore trained = TrainingCore.trainModel(events, List.of(".php"));
        @SuppressWarnings("unchecked")
        IsolationForestCore.Model before = (IsolationForestCore.Model)
                ((Map<String, Object>) trained.payload().get("isolation_forest")).get("model");
        java.nio.file.Path file = Files.createTempFile("aiwaf-portable-roundtrip-", ".json");
        assertTrue(ModelArtifactIoCore.save(trained, file.toString()));
        TrainedModelCore loaded = ModelArtifactIoCore.load(file.toString());
        assertNotNull(loaded);
        @SuppressWarnings("unchecked")
        IsolationForestCore.Model after = (IsolationForestCore.Model)
                ((Map<String, Object>) loaded.payload().get("isolation_forest")).get("model");
        double[] row = {12, 0, 0.004, 0, 1, 0};
        assertEquals(IsolationForestCore.score(before, row), IsolationForestCore.score(after, row), 1e-12);
    }
}
