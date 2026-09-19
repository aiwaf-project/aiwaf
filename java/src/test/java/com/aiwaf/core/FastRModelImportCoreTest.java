package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FastRModelImportCoreTest {

    @Test
    void imports_neutral_tree_json_into_java_model() throws Exception {
        TrainedModelCore trained = FastRModelImportCore.parse(goldenJson());
        @SuppressWarnings("unchecked")
        Map<String, Object> iforest = (Map<String, Object>) trained.payload().get("isolation_forest");
        IsolationForestCore.Model model = (IsolationForestCore.Model) iforest.get("model");

        assertEquals("fastr_aiwaf", iforest.get("backend"));
        assertEquals(2, model.sampleSize());
        assertEquals(0.5, IsolationForestCore.score(model, new double[]{5.0}), 1e-12);
        assertEquals(0.5, IsolationForestCore.score(model, new double[]{15.0}), 1e-12);
    }

    @Test
    void rejects_unsupported_schema() {
        String bad = goldenJson().replace("\"iforest-v1\"", "\"iforest-v2\"");
        assertThrows(IllegalArgumentException.class, () -> FastRModelImportCore.parse(bad));
    }

    @Test
    void preserves_six_feature_metadata_and_static_keywords() throws Exception {
        String json = goldenJson()
                .replace("\"feature_names\":[\"x\"]", "\"feature_names\":[\"path_len\",\"kw_hits\",\"resp_time\",\"status_idx\",\"burst_count\",\"total_404\"]")
                .replace("\"sample_size\":2,", "\"static_keywords\":[],\"sample_size\":2,")
                .replace("\"created_at_epoch_ms\":1760000000000", "\"feature_schema\":\"python-six-v1\",\"created_at_epoch_ms\":1760000000000");
        TrainedModelCore trained = FastRModelImportCore.parse(json);
        @SuppressWarnings("unchecked")
        Map<String, Object> iforest = (Map<String, Object>) trained.payload().get("isolation_forest");
        @SuppressWarnings("unchecked")
        Map<String, Object> metadata = (Map<String, Object>) trained.payload().get("metadata");

        assertEquals("python-six-v1", metadata.get("feature_schema"));
        assertEquals(List.of(), iforest.get("static_keywords"));
        assertEquals(6, ((List<?>) iforest.get("feature_names")).size());
    }

    @Test
    void rejects_feature_index_outside_feature_names() {
        String bad = goldenJson().replace("\"feature\":0", "\"feature\":1");
        assertThrows(IllegalArgumentException.class, () -> FastRModelImportCore.parse(bad));
    }

    @Test
    void imported_model_artifact_remains_loadable_by_runtime_io() throws Exception {
        TrainedModelCore trained = FastRModelImportCore.parse(goldenJson());
        java.nio.file.Path file = java.nio.file.Files.createTempFile("aiwaf-fastr-import-", ".bin");

        assertTrue(ModelArtifactIoCore.save(trained, file.toString()));
        assertEquals("isolation-forest", ModelArtifactIoCore.load(file.toString()).modelType());
    }

    static String goldenJson() {
        return """
                {
                  "model_type":"isolation-forest",
                  "model_schema":"iforest-v1",
                  "backend":"fastr_aiwaf",
                  "feature_names":["x"],
                  "sample_size":2,
                  "trees":[
                    {
                      "feature_subset":[0],
                      "root":{
                        "leaf":false,
                        "feature":0,
                        "split":10.0,
                        "left":{"leaf":true,"leaf_size":1},
                        "right":{"leaf":true,"leaf_size":1}
                      }
                    }
                  ],
                  "contamination":0.05,
                  "threshold":0.4,
                  "anomaly_count":1,
                  "samples":2,
                  "avg_response_time_ms":3.0,
                  "status_counts":{"200":2},
                  "behavior":{},
                  "metadata":{"created_at_epoch_ms":1760000000000}
                }
                """;
    }
}
