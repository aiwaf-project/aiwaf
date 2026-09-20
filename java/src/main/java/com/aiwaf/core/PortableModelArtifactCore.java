package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Converts Java isolation forests to and from the JSON artifact consumed by aiwaf-rust. */
public final class PortableModelArtifactCore {
    public static final String ARTIFACT_SCHEMA = "aiwaf-model-v1";
    public static final String PORTABLE_MODEL_TYPE = "aiwaf_rust.IsolationForest";
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private PortableModelArtifactCore() {}

    public static Map<String, Object> toPortableArtifact(TrainedModelCore trained) {
        if (trained == null || trained.payload() == null) {
            throw new IllegalArgumentException("model payload is required");
        }
        Object iforestObject = trained.payload().get("isolation_forest");
        if (!(iforestObject instanceof Map<?, ?> rawForest)) {
            throw new IllegalArgumentException("isolation_forest payload is required");
        }
        Map<String, Object> forest = stringMap(rawForest);
        if (!(forest.get("model") instanceof IsolationForestCore.Model model)) {
            throw new IllegalArgumentException("Java isolation forest model is required");
        }
        List<String> featureNames = stringList(forest.get("feature_names"));
        if (featureNames.isEmpty()) {
            throw new IllegalArgumentException("feature_names are required");
        }

        Map<String, Object> metadata = stringMap(trained.payload().get("metadata"));
        long createdAt = number(metadata.get("created_at_epoch_ms"), System.currentTimeMillis()).longValue();
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("artifact_schema", ARTIFACT_SCHEMA);
        root.put("model_type", PORTABLE_MODEL_TYPE);
        root.put("model_backend", "aiwaf_rust");
        root.put("model_state", toRustState(model, featureNames));
        root.put("created_at", Instant.ofEpochMilli(createdAt).toString());
        root.put("feature_count", featureNames.size());
        root.put("samples_count", number(trained.payload().get("samples"), 0).intValue());
        root.put("framework", "java");
        root.put("backend", "rust");
        root.put("producer_backend", "aiwaf_java");
        root.put("feature_names", featureNames);

        Map<String, Object> portablePayload = new LinkedHashMap<>();
        copyJsonValue(trained.payload(), portablePayload, "avg_response_time_ms");
        copyJsonValue(trained.payload(), portablePayload, "status_counts");
        copyJsonValue(trained.payload(), portablePayload, "samples");
        copyJsonValue(trained.payload(), portablePayload, "behavior");
        portablePayload.put("metadata", metadata);
        Map<String, Object> forestMetadata = new LinkedHashMap<>(forest);
        forestMetadata.remove("model");
        portablePayload.put("isolation_forest", forestMetadata);
        root.put("payload", portablePayload);
        return root;
    }

    public static TrainedModelCore parse(String json) throws IOException {
        if (json == null || json.isBlank()) throw new IllegalArgumentException("model JSON must not be blank");
        JsonNode root = MAPPER.readTree(json);
        String modelType = root.path("model_type").asText("");
        if (!PORTABLE_MODEL_TYPE.equals(modelType)) {
            throw new IllegalArgumentException("unsupported model_type=" + modelType);
        }
        JsonNode state = root.path("model_state");
        if (!state.isObject()) throw new IllegalArgumentException("model_state must be an object");

        int featureCount = positiveInt(first(state, "nFeaturesIn_", "n_features_in_"),
                root.path("feature_count").asInt(0));
        List<String> featureNames = textList(root.path("feature_names"));
        if (featureNames.isEmpty()) featureNames = defaultFeatureNames(featureCount);
        if (featureCount <= 0) featureCount = featureNames.size();
        if (featureNames.size() != featureCount) {
            throw new IllegalArgumentException("feature_names size does not match model state");
        }

        int sampleSize = positiveInt(first(state, "maxSamples_", "max_samples_"), 0);
        JsonNode treeNodes = state.path("trees");
        if (!treeNodes.isArray() || treeNodes.isEmpty()) {
            throw new IllegalArgumentException("model_state.trees must be a non-empty array");
        }
        JsonNode featureSets = first(state, "estimatorsFeatures", "estimators_features");
        List<IsolationForestCore.IsolationTree> trees = new ArrayList<>();
        for (int i = 0; i < treeNodes.size(); i++) {
            JsonNode treeNode = treeNodes.get(i);
            int[] subset = featureSets.isArray() && i < featureSets.size()
                    ? intArray(featureSets.get(i), featureCount)
                    : allFeatures(featureCount);
            trees.add(new IsolationForestCore.IsolationTree(readRustNode(treeNode, featureCount), subset));
        }
        IsolationForestCore.Model model = new IsolationForestCore.Model(trees, sampleSize);

        Map<String, Object> payload = objectMap(root.path("payload"));
        Map<String, Object> forest = stringMap(payload.get("isolation_forest"));
        forest.put("model", model);
        forest.put("trees", trees.size());
        forest.put("sample_size", sampleSize);
        forest.putIfAbsent("threshold", 0.5);
        forest.putIfAbsent("contamination", scalar(first(state, "contamination")));
        forest.put("feature_names", featureNames);
        forest.putIfAbsent("static_keywords", List.of());
        forest.putIfAbsent("backend", String.valueOf(root.path("producer_backend").asText("aiwaf_rust")));
        payload.put("isolation_forest", forest);
        payload.putIfAbsent("samples", root.path("samples_count").asInt(0));
        payload.putIfAbsent("avg_response_time_ms", 0.0);
        payload.putIfAbsent("status_counts", Map.of());
        payload.putIfAbsent("behavior", Map.of());

        Map<String, Object> metadata = stringMap(payload.get("metadata"));
        metadata.putIfAbsent("model_schema", "iforest-v1");
        metadata.putIfAbsent("feature_schema", featureCount == 6 ? "python-six-v1" : "portable-v1");
        metadata.putIfAbsent("model_backend", root.path("producer_backend").asText("aiwaf_rust"));
        metadata.putIfAbsent("imported_from", "portable_json");
        payload.put("metadata", metadata);
        return new TrainedModelCore("isolation-forest", "1", payload);
    }

    private static Map<String, Object> toRustState(IsolationForestCore.Model model, List<String> featureNames) {
        List<Map<String, Object>> trees = new ArrayList<>();
        List<List<Integer>> featureSets = new ArrayList<>();
        int maxFeatures = 1;
        for (IsolationForestCore.IsolationTree tree : model.trees()) {
            int maxDepth = maxDepth(tree.root(), 0);
            trees.add(writeRustNode(tree.root(), 0, maxDepth));
            List<Integer> subset = new ArrayList<>();
            for (int feature : tree.featureSubset()) subset.add(feature);
            featureSets.add(subset);
            maxFeatures = Math.max(maxFeatures, subset.size());
        }
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("nEstimators", model.trees().size());
        state.put("maxSamples", model.sampleSize());
        state.put("contamination", "auto");
        state.put("maxFeatures", maxFeatures);
        state.put("bootstrap", false);
        state.put("randomState", null);
        state.put("verbose", 0);
        state.put("warmStart", false);
        state.put("maxSamples_", model.sampleSize());
        state.put("maxFeatures_", maxFeatures);
        state.put("offset_", -0.5);
        state.put("nFeaturesIn_", featureNames.size());
        state.put("trees", trees);
        state.put("estimatorsFeatures", featureSets);
        return state;
    }

    private static Map<String, Object> writeRustNode(IsolationForestCore.Node node, int depth, int maxDepth) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("depth", depth);
        out.put("maxDepth", maxDepth);
        out.put("splitAttr", node.leaf() ? null : node.feature());
        out.put("splitValue", node.leaf() ? null : node.split());
        out.put("size", subtreeSize(node));
        out.put("left", node.left() == null ? null : writeRustNode(node.left(), depth + 1, maxDepth));
        out.put("right", node.right() == null ? null : writeRustNode(node.right(), depth + 1, maxDepth));
        return out;
    }

    private static IsolationForestCore.Node readRustNode(JsonNode node, int featureCount) {
        if (node == null || !node.isObject()) throw new IllegalArgumentException("tree node must be an object");
        JsonNode splitAttr = first(node, "splitAttr", "split_attr");
        JsonNode splitValue = first(node, "splitValue", "split_value");
        JsonNode left = node.path("left");
        JsonNode right = node.path("right");
        if (splitAttr.isNull() || splitAttr.isMissingNode() || left.isNull() || right.isNull()) {
            int size = Math.max(0, node.path("size").asInt(0));
            return new IsolationForestCore.Node(true, size, -1, 0.0, null, null);
        }
        int feature = splitAttr.asInt(-1);
        double split = splitValue.asDouble(Double.NaN);
        if (feature < 0 || feature >= featureCount || !Double.isFinite(split)) {
            throw new IllegalArgumentException("invalid tree split");
        }
        return new IsolationForestCore.Node(
                false,
                0,
                feature,
                split,
                readRustNode(left, featureCount),
                readRustNode(right, featureCount)
        );
    }

    private static int maxDepth(IsolationForestCore.Node node, int depth) {
        if (node == null || node.leaf()) return depth;
        return Math.max(maxDepth(node.left(), depth + 1), maxDepth(node.right(), depth + 1));
    }

    private static int subtreeSize(IsolationForestCore.Node node) {
        if (node == null) return 0;
        if (node.leaf()) return Math.max(0, node.leafSize());
        return subtreeSize(node.left()) + subtreeSize(node.right());
    }

    private static int[] intArray(JsonNode node, int featureCount) {
        if (!node.isArray() || node.isEmpty()) return allFeatures(featureCount);
        int[] out = new int[node.size()];
        for (int i = 0; i < node.size(); i++) {
            out[i] = node.get(i).asInt(-1);
            if (out[i] < 0 || out[i] >= featureCount) {
                throw new IllegalArgumentException("invalid estimator feature index");
            }
        }
        return out;
    }

    private static int[] allFeatures(int featureCount) {
        int[] out = new int[Math.max(0, featureCount)];
        for (int i = 0; i < out.length; i++) out[i] = i;
        return out;
    }

    private static List<String> defaultFeatureNames(int featureCount) {
        if (featureCount == FeaturesCore.PYTHON_FEATURE_NAMES.size()) {
            return FeaturesCore.PYTHON_FEATURE_NAMES;
        }
        List<String> out = new ArrayList<>();
        for (int i = 0; i < featureCount; i++) out.add("feature_" + i);
        return out;
    }

    private static JsonNode first(JsonNode node, String... names) {
        if (node == null) return MAPPER.missingNode();
        for (String name : names) {
            JsonNode value = node.get(name);
            if (value != null) return value;
        }
        return MAPPER.missingNode();
    }

    private static int positiveInt(JsonNode value, int fallback) {
        int result = value == null || value.isMissingNode() ? fallback : value.asInt(fallback);
        if (result <= 0) throw new IllegalArgumentException("model state requires a positive size");
        return result;
    }

    private static List<String> textList(JsonNode node) {
        List<String> out = new ArrayList<>();
        if (node == null || !node.isArray()) return out;
        for (JsonNode item : node) if (!item.asText("").isBlank()) out.add(item.asText());
        return out;
    }

    private static List<String> stringList(Object value) {
        List<String> out = new ArrayList<>();
        if (value instanceof Iterable<?> values) {
            for (Object item : values) if (item != null && !String.valueOf(item).isBlank()) out.add(String.valueOf(item));
        }
        return out;
    }

    private static Map<String, Object> objectMap(JsonNode node) {
        if (node == null || !node.isObject()) return new HashMap<>();
        return MAPPER.convertValue(node, MAPPER.getTypeFactory().constructMapType(HashMap.class, String.class, Object.class));
    }

    private static Map<String, Object> stringMap(Object value) {
        Map<String, Object> out = new HashMap<>();
        if (!(value instanceof Map<?, ?> map)) return out;
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (entry.getKey() != null) out.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return out;
    }

    private static Number number(Object value, Number fallback) {
        return value instanceof Number number ? number : fallback;
    }

    private static Object scalar(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) return "auto";
        if (node.isNumber()) return node.numberValue();
        if (node.isTextual()) return node.asText();
        if (node.isObject()) {
            if (node.has("Fixed")) return node.path("Fixed").numberValue();
            if (node.has("Auto")) return "auto";
        }
        return "auto";
    }

    private static void copyJsonValue(Map<String, Object> source, Map<String, Object> target, String key) {
        if (source.containsKey(key)) target.put(key, source.get(key));
    }
}
