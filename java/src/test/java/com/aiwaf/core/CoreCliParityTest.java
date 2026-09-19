package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class CoreCliParityTest {

    @TempDir
    Path tempDir;

    @Test
    void reset_and_list_default_targets_are_valid() {
        assertEquals(0, CoreCli.main(new String[]{"reset"}));
        assertEquals(0, CoreCli.main(new String[]{"list"}));
    }

    @Test
    void reset_and_list_reject_invalid_targets() {
        assertEquals(2, CoreCli.main(new String[]{"reset", "--targets", "bad"}));
        assertEquals(2, CoreCli.main(new String[]{"list", "--targets", "all,features"}));
        assertEquals(2, CoreCli.main(new String[]{"list", "--targets", ","}));
    }

    @Test
    void replay_requires_cases_option() {
        assertEquals(2, CoreCli.main(new String[]{"replay"}));
        assertEquals(2, CoreCli.main(new String[]{"replay", "--cases"}));
    }

    @Test
    void analyze_behavior_requires_events_option() {
        assertEquals(2, CoreCli.main(new String[]{"analyze-behavior"}));
        assertEquals(2, CoreCli.main(new String[]{"analyze-behavior", "--events"}));
    }

    @Test
    void analyze_request_requires_request_option() {
        assertEquals(2, CoreCli.main(new String[]{"analyze-request"}));
        assertEquals(2, CoreCli.main(new String[]{"analyze-request", "--request"}));
    }

    @Test
    void analyze_request_with_invalid_json_returns_error() throws Exception {
        Path reqJson = tempDir.resolve("bad.json");
        Files.writeString(reqJson, "{", StandardCharsets.UTF_8);
        assertEquals(1, CoreCli.main(new String[]{"analyze-request", "--request", reqJson.toString()}));
    }

    @Test
    void train_model_java_backend_writes_loadable_artifact() throws Exception {
        Path events = writeEvents();
        Path out = tempDir.resolve("java-model.bin");

        assertEquals(0, CoreCli.main(new String[]{
                "train-model",
                "--events", events.toString(),
                "--out", out.toString(),
                "--backend", "java"
        }));
        TrainedModelCore model = ModelArtifactIoCore.load(out.toString());
        assertNotNull(model);
        Map<?, ?> iforest = (Map<?, ?>) model.payload().get("isolation_forest");
        assertEquals(TrainingCore.FLASK_STATIC_KEYWORDS.size(),
                ((List<?>) iforest.get("static_keywords")).size());
    }

    @Test
    void train_model_java_backend_accepts_empty_keyword_override() throws Exception {
        Path out = tempDir.resolve("no-keywords-model.bin");
        assertEquals(0, CoreCli.main(new String[]{
                "train-model",
                "--events", writeEvents().toString(),
                "--out", out.toString(),
                "--backend", "java",
                "--static-keywords", ""
        }));
        TrainedModelCore model = ModelArtifactIoCore.load(out.toString());
        assertNotNull(model);
        Map<?, ?> iforest = (Map<?, ?>) model.payload().get("isolation_forest");
        assertEquals(List.of(), iforest.get("static_keywords"));
    }

    @Test
    void train_model_fastr_backend_imports_script_json() throws Exception {
        Path events = writeEvents();
        Path out = tempDir.resolve("fastr-model.bin");
        FakeFastR fake = writeFakeFastRCommand();

        assertEquals(0, CoreCli.main(new String[]{
                "train-model",
                "--events", events.toString(),
                "--out", out.toString(),
                "--backend", "fastr",
                "--fastr-command", fake.command().toString(),
                "--fastr-script", fake.script().toString()
        }));
        assertNotNull(ModelArtifactIoCore.load(out.toString()));
    }

    @Test
    void train_model_fastr_backend_falls_back_to_java_when_command_is_missing() throws Exception {
        Path events = writeEvents();
        Path out = tempDir.resolve("fallback-model.bin");

        assertEquals(0, CoreCli.main(new String[]{
                "train-model",
                "--events", events.toString(),
                "--out", out.toString(),
                "--backend", "fastr",
                "--fastr-command", tempDir.resolve("missing-fastr-command").toString()
        }));
        assertNotNull(ModelArtifactIoCore.load(out.toString()));
    }

    @Test
    void train_model_fastr_backend_falls_back_to_java_when_script_fails() throws Exception {
        Path events = writeEvents();
        Path out = tempDir.resolve("failed-fastr-model.bin");
        Path failingCommand = writeFailingFastRCommand();

        assertEquals(0, CoreCli.main(new String[]{
                "train-model",
                "--events", events.toString(),
                "--out", out.toString(),
                "--backend", "fastr",
                "--fastr-command", failingCommand.toString(),
                "--fastr-script", "ignored"
        }));
        assertNotNull(ModelArtifactIoCore.load(out.toString()));
    }

    @Test
    void train_model_rejects_missing_options_and_unknown_backend() throws Exception {
        Path events = writeEvents();
        assertEquals(2, CoreCli.main(new String[]{"train-model"}));
        assertEquals(2, CoreCli.main(new String[]{
                "train-model",
                "--events", events.toString(),
                "--out", tempDir.resolve("model.bin").toString(),
                "--backend", "unknown"
        }));
    }

    @Test
    void unknown_command_returns_usage_error() {
        assertEquals(2, CoreCli.main(new String[]{"unknown"}));
    }

    private Path writeEvents() throws Exception {
        Path events = tempDir.resolve("events.json");
        Files.writeString(events, """
                [
                  {
                    "ip": "1.1.1.1",
                    "method": "GET",
                    "path": "/home",
                    "status_code": 200,
                    "response_time_ms": 10.0,
                    "timestamp": "2026-01-01T00:00:00",
                    "known_path": true,
                    "exempt_path": false
                  },
                  {
                    "ip": "1.1.1.2",
                    "method": "POST",
                    "path": "/wp-admin/shell",
                    "status_code": 404,
                    "response_time_ms": 15.0,
                    "timestamp": "2026-01-01T00:00:01",
                    "known_path": false,
                    "exempt_path": false
                  }
                ]
                """, StandardCharsets.UTF_8);
        return events;
    }

    private FakeFastR writeFakeFastRCommand() throws Exception {
        Path sourceJson = tempDir.resolve("fake-fastr-model.json");
        Files.writeString(sourceJson, FastRModelImportCoreTest.goldenJson(), StandardCharsets.UTF_8);
        if (isWindows()) {
            Path command = tempDir.resolve("fake-fastr.cmd");
            Files.writeString(command, """
                    @echo off
                    set source=%~1
                    shift
                    set out=
                    :loop
                    if "%~1"=="" goto done
                    if "%~1"=="--out" (
                      set out=%~2
                      shift
                      shift
                      goto loop
                    )
                    shift
                    goto loop
                    :done
                    copy /Y "%source%" "%out%" >NUL
                    exit /b %ERRORLEVEL%
                    """, StandardCharsets.UTF_8);
            return new FakeFastR(command, sourceJson);
        }

        Path command = tempDir.resolve("fake-fastr.sh");
        Files.writeString(command, """
                #!/bin/sh
                source="$1"
                shift
                out=""
                while [ "$#" -gt 0 ]; do
                  case "$1" in
                    --out) out="$2"; shift 2 ;;
                    *) shift ;;
                  esac
                done
                cp "$source" "$out"
                """, StandardCharsets.UTF_8);
        command.toFile().setExecutable(true);
        return new FakeFastR(command, sourceJson);
    }

    private Path writeFailingFastRCommand() throws Exception {
        if (isWindows()) {
            Path command = tempDir.resolve("failing-fastr.cmd");
            Files.writeString(command, """
                    @echo off
                    exit /b 7
                    """, StandardCharsets.UTF_8);
            return command;
        }

        Path command = tempDir.resolve("failing-fastr.sh");
        Files.writeString(command, """
                #!/bin/sh
                exit 7
                """, StandardCharsets.UTF_8);
        command.toFile().setExecutable(true);
        return command;
    }

    private static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win");
    }

    private record FakeFastR(Path command, Path script) {}
}
