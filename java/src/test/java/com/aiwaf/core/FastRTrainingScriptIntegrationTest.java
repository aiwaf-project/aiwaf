package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class FastRTrainingScriptIntegrationTest {

    @TempDir
    Path tempDir;

    @Test
    void fastr_script_trains_importable_isolation_forest_when_r_is_available() throws Exception {
        assumeTrue(commandSucceeds(List.of("Rscript", "--version")), "Rscript is not available");
        assumeTrue(commandSucceeds(List.of(
                "Rscript",
                "-e",
                "quit(status=ifelse(requireNamespace('jsonlite', quietly=TRUE), 0, 2))"
        )), "R package jsonlite is not available");

        Path out = tempDir.resolve("fastr-model.json");
        ProcessBuilder builder = new ProcessBuilder(
                "Rscript",
                "scripts/fastr/train_iforest.R",
                "--events",
                "src/test/resources/events.json",
                "--out",
                out.toString()
        ).redirectErrorStream(true);
        addLocalRLibrary(builder);
        Process process = builder.start();

        assertEquals(0, process.waitFor());
        assertTrue(Files.isRegularFile(out));

        TrainedModelCore model = FastRModelImportCore.load(out);
        @SuppressWarnings("unchecked")
        Map<String, Object> iforest = (Map<String, Object>) model.payload().get("isolation_forest");
        @SuppressWarnings("unchecked")
        Map<String, Object> metadata = (Map<String, Object>) model.payload().get("metadata");

        assertEquals("isolation-forest", model.modelType());
        assertEquals("fastr_aiwaf", iforest.get("backend"));
        assertEquals("fastr_aiwaf", metadata.get("model_backend"));
        assertEquals("python-six-v1", metadata.get("feature_schema"));
        assertEquals(List.of("path_len", "kw_hits", "resp_time", "status_idx", "burst_count", "total_404"),
                iforest.get("feature_names"));
        assertEquals(TrainingCore.FLASK_STATIC_KEYWORDS.size(),
                ((List<?>) iforest.get("static_keywords")).size());
        assertTrue(((List<?>) iforest.get("static_keywords")).contains("shell"));
        assertEquals(12.5, (double) model.payload().get("avg_response_time_ms"), 1e-9);
    }

    private static boolean commandSucceeds(List<String> command) {
        try {
            ProcessBuilder builder = new ProcessBuilder(command)
                    .redirectErrorStream(true)
                    .redirectOutput(ProcessBuilder.Redirect.DISCARD);
            addLocalRLibrary(builder);
            Process process = builder.start();
            return process.waitFor() == 0;
        } catch (Exception ex) {
            return false;
        }
    }

    private static void addLocalRLibrary(ProcessBuilder builder) {
        Path localLib = Path.of(".r-lib").toAbsolutePath().normalize();
        if (Files.isDirectory(localLib)) {
            builder.environment().put("R_LIBS_USER", localLib.toString());
        }
    }
}
