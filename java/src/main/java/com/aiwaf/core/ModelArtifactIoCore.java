package com.aiwaf.core;

import java.io.FileInputStream;
import java.io.ObjectOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.logging.Logger;

public final class ModelArtifactIoCore {
    private static final Logger LOG = Logger.getLogger(ModelArtifactIoCore.class.getName());

    private ModelArtifactIoCore() {}

    public static boolean save(TrainedModelCore model, String path) {
        if (model == null || path == null || path.isBlank()) {
            return false;
        }
        try {
            Path p = Path.of(path);
            Map<String, Object> portable = PortableModelArtifactCore.toPortableArtifact(model);
            SecureFiles.writeAtomically(p, output ->
                    new com.fasterxml.jackson.databind.ObjectMapper().writeValue(output, portable));
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    public static TrainedModelCore load(String path) {
        if (path == null || path.isBlank()) {
            return null;
        }
        try {
            Path p = Path.of(path);
            if (!Files.exists(p)) {
                return null;
            }
            SecureFiles.rejectSymbolicLinks(p);
            if (!SecureFiles.verifySignature(p)) {
                LOG.warning("Ignoring AIWAF model artifact with invalid or missing signature");
                return null;
            }
            TrainedModelCore model = looksLikeJson(p) ? loadPortable(p) : loadLegacy(p);
            if (model == null) return null;
            TrainedModelCore migrated = ModelArtifactMigrationCore.migrate(model);
            if (!isCompatible(migrated)) return null;
            return migrated;
        } catch (Exception ex) {
            LOG.warning("Unable to load AIWAF model artifact: " + ex.getClass().getSimpleName());
        }
        return null;
    }

    static boolean saveLegacy(TrainedModelCore model, String path) {
        try {
            SecureFiles.writeAtomically(Path.of(path), output -> {
                try (ObjectOutputStream out = new ObjectOutputStream(output)) {
                    out.writeObject(model);
                }
            });
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    private static TrainedModelCore loadPortable(Path path) throws Exception {
        return PortableModelArtifactCore.parse(Files.readString(path, StandardCharsets.UTF_8));
    }

    private static TrainedModelCore loadLegacy(Path path) throws Exception {
        try (var in = SafeObjectInputStreams.open(
                new FileInputStream(path.toFile()), SafeObjectInputStreams.Profile.MODEL)) {
            Object obj = in.readObject();
            return obj instanceof TrainedModelCore model ? model : null;
        }
    }

    private static boolean looksLikeJson(Path path) throws Exception {
        try (var input = Files.newInputStream(path)) {
            int value;
            while ((value = input.read()) >= 0) {
                if (!Character.isWhitespace(value)) return value == '{';
            }
            return false;
        }
    }

    private static boolean isCompatible(TrainedModelCore model) {
        if (model.modelType() == null || !model.modelType().equals("isolation-forest")) {
            LOG.warning("Ignoring incompatible AIWAF model artifact: unsupported modelType=" + model.modelType());
            return false;
        }
        if (model.payload() == null) {
            LOG.warning("Ignoring incompatible AIWAF model artifact: missing payload");
            return false;
        }
        Object ifObj = model.payload().get("isolation_forest");
        if (!(ifObj instanceof Map<?, ?> ifMap)) {
            LOG.warning("Ignoring incompatible AIWAF model artifact: missing isolation_forest payload");
            return false;
        }
        if (!(ifMap.get("model") instanceof IsolationForestCore.Model)) {
            LOG.warning("Ignoring incompatible AIWAF model artifact: missing serialized isolation forest model");
            return false;
        }
        Object metadataObj = model.payload().get("metadata");
        if (metadataObj instanceof Map<?, ?> metadata) {
            Object schema = metadata.get("model_schema");
            if (schema != null && !"iforest-v1".equals(String.valueOf(schema))) {
                LOG.warning("Ignoring incompatible AIWAF model artifact: unsupported model_schema=" + schema);
                return false;
            }
        }
        return true;
    }
}
