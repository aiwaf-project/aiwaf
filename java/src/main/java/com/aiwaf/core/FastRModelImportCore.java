package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class FastRModelImportCore {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private FastRModelImportCore() {}

    public static TrainedModelCore load(Path path) throws IOException {
        if (path == null) {
            throw new IllegalArgumentException("path must not be null");
        }
        return parse(Files.readString(path));
    }

    public static TrainedModelCore parse(String json) throws IOException {
        if (json == null || json.isBlank()) {
            throw new IllegalArgumentException("model JSON must not be blank");
        }
        JsonNode root = MAPPER.readTree(json);
        validateText(root, "model_type", "isolation-forest");
        validateText(root, "model_schema", "iforest-v1");

        List<String> featureNames = readFeatureNames(root.path("feature_names"));
        int sampleSize = positiveInt(root, "sample_size");
        List<IsolationForestCore.IsolationTree> trees = readTrees(root.path("trees"), featureNames.size());
        IsolationForestCore.Model model = new IsolationForestCore.Model(trees, sampleSize);

        Map<String, Object> isolationForest = new HashMap<>();
        isolationForest.put("backend", root.path("backend").asText("fastr_aiwaf"));
        isolationForest.put("trees", trees.size());
        isolationForest.put("sample_size", sampleSize);
        isolationForest.put("contamination", root.path("contamination").asDouble(0.05));
        isolationForest.put("threshold", root.path("threshold").asDouble(1.0));
        isolationForest.put("anomaly_count", root.path("anomaly_count").asInt(0));
        isolationForest.put("feature_names", featureNames);
        JsonNode staticKeywords = root.path("static_keywords");
        if (staticKeywords.isArray()) {
            List<String> keywords = new ArrayList<>();
            for (JsonNode keyword : staticKeywords) {
                keywords.add(keyword.asText());
            }
            isolationForest.put("static_keywords", keywords);
        }
        isolationForest.put("model", model);

        Map<String, Object> metadata = new HashMap<>();
        metadata.put("model_backend", "fastr_aiwaf");
        metadata.put("model_schema", "iforest-v1");
        metadata.put("imported_from", "fastr_json");
        JsonNode metadataNode = root.path("metadata");
        if (metadataNode.isObject()) {
            metadataNode.fields().forEachRemaining(e -> metadata.put(e.getKey(), jsonValue(e.getValue())));
        }

        Map<String, Object> payload = new HashMap<>();
        payload.put("avg_response_time_ms", root.path("avg_response_time_ms").asDouble(0.0));
        payload.put("status_counts", readObjectMap(root.path("status_counts")));
        payload.put("samples", root.path("samples").asInt(0));
        payload.put("behavior", readObjectMap(root.path("behavior")));
        payload.put("isolation_forest", isolationForest);
        payload.put("metadata", metadata);
        return new TrainedModelCore("isolation-forest", "1", payload);
    }

    private static void validateText(JsonNode root, String field, String expected) {
        String actual = root.path(field).asText("");
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException("unsupported " + field + "=" + actual);
        }
    }

    private static int positiveInt(JsonNode root, String field) {
        int value = root.path(field).asInt(0);
        if (value <= 0) {
            throw new IllegalArgumentException(field + " must be positive");
        }
        return value;
    }

    private static List<String> readFeatureNames(JsonNode node) {
        if (!node.isArray() || node.isEmpty()) {
            throw new IllegalArgumentException("feature_names must be a non-empty array");
        }
        List<String> out = new ArrayList<>();
        for (JsonNode item : node) {
            String value = item.asText("");
            if (value.isBlank()) {
                throw new IllegalArgumentException("feature_names must not contain blanks");
            }
            out.add(value);
        }
        return out;
    }

    private static List<IsolationForestCore.IsolationTree> readTrees(JsonNode node, int featureCount) {
        if (!node.isArray() || node.isEmpty()) {
            throw new IllegalArgumentException("trees must be a non-empty array");
        }
        List<IsolationForestCore.IsolationTree> out = new ArrayList<>();
        for (JsonNode tree : node) {
            int[] featureSubset = readIntArray(tree.path("feature_subset"), "feature_subset", featureCount);
            IsolationForestCore.Node root = readNode(tree.path("root"), featureCount);
            out.add(new IsolationForestCore.IsolationTree(root, featureSubset));
        }
        return out;
    }

    private static IsolationForestCore.Node readNode(JsonNode node, int featureCount) {
        if (!node.isObject()) {
            throw new IllegalArgumentException("tree node must be an object");
        }
        boolean leaf = node.path("leaf").asBoolean(false);
        if (leaf) {
            int leafSize = node.path("leaf_size").asInt(0);
            if (leafSize < 0) {
                throw new IllegalArgumentException("leaf_size must be non-negative");
            }
            return new IsolationForestCore.Node(true, leafSize, -1, 0.0, null, null);
        }
        int feature = node.path("feature").asInt(-1);
        double split = node.path("split").asDouble(Double.NaN);
        if (feature < 0 || feature >= featureCount || !Double.isFinite(split)) {
            throw new IllegalArgumentException("branch nodes require feature and finite split");
        }
        return new IsolationForestCore.Node(
                false,
                0,
                feature,
                split,
                readNode(node.path("left"), featureCount),
                readNode(node.path("right"), featureCount)
        );
    }

    private static int[] readIntArray(JsonNode node, String field, int featureCount) {
        if (!node.isArray() || node.isEmpty()) {
            throw new IllegalArgumentException(field + " must be a non-empty array");
        }
        int[] out = new int[node.size()];
        for (int i = 0; i < node.size(); i++) {
            out[i] = node.get(i).asInt(-1);
            if (out[i] < 0 || out[i] >= featureCount) {
                throw new IllegalArgumentException(field + " must contain valid feature indexes");
            }
        }
        return out;
    }

    private static Map<String, Object> readObjectMap(JsonNode node) {
        if (!node.isObject()) {
            return new HashMap<>();
        }
        Map<String, Object> out = new HashMap<>();
        node.fields().forEachRemaining(e -> out.put(e.getKey(), jsonValue(e.getValue())));
        return out;
    }

    private static Object jsonValue(JsonNode node) {
        if (node.isObject()) {
            return readObjectMap(node);
        }
        if (node.isArray()) {
            List<Object> out = new ArrayList<>();
            for (JsonNode item : node) {
                out.add(jsonValue(item));
            }
            return out;
        }
        if (node.isIntegralNumber()) {
            return node.longValue();
        }
        if (node.isFloatingPointNumber()) {
            return node.doubleValue();
        }
        if (node.isBoolean()) {
            return node.booleanValue();
        }
        if (node.isNull()) {
            return null;
        }
        return node.asText();
    }
}
