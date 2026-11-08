package com.aiwaf.cli;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AiwafConsoleLogicTest {

    private static final String VULNERABLE = """
            @RestController
            class App {
                @PostMapping("/login")
                String view(@RequestParam String value) throws Exception {
                    Runtime.getRuntime().exec(value);
                    return "ok";
                }
            }
            """;

    @Test
    void audit_returns_exit_codes_and_writes_reports(@org.junit.jupiter.api.io.TempDir java.nio.file.Path directory) throws Exception {
        java.nio.file.Files.writeString(directory.resolve("App.java"), VULNERABLE);
        String root = directory.toString();
        java.nio.file.Path report = directory.resolve("report.json");

        assertEquals(0, AiwafConsole.handleAudit(new String[]{"audit", root, "--output", report.toString()}));
        assertTrue(java.nio.file.Files.readString(report).contains("found 1 possible"));
        assertEquals(1, AiwafConsole.handleAudit(new String[]{"audit", root, "--fail-on-findings", "--output", report.toString()}));
        assertEquals(0, AiwafConsole.handleAudit(new String[]{"audit", root, "--format", "json", "--output", report.toString()}));
        assertTrue(java.nio.file.Files.readString(report).contains("\"language\" : \"java\""));
        assertEquals(2, AiwafConsole.handleAudit(new String[]{"audit", root, "--format", "xml"}));
        assertEquals(2, AiwafConsole.handleAudit(new String[]{"audit", root, "--output"}));
        assertEquals(2, AiwafConsole.handleAudit(new String[]{"audit", directory.resolve("missing").toString()}));
    }

    @Test
    void audit_saves_findings_to_a_manifest(@org.junit.jupiter.api.io.TempDir java.nio.file.Path directory) throws Exception {
        java.nio.file.Files.writeString(directory.resolve("App.java"), VULNERABLE);
        java.nio.file.Path manifest = directory.resolve("paths.json");
        String root = directory.toString();
        java.nio.file.Path report = directory.resolve("report.txt");
        assertEquals(2, AiwafConsole.handleAudit(new String[]{"audit", root, "--output", report.toString(), "--manifest", manifest.toString()}));
        assertEquals(2, AiwafConsole.handleAudit(new String[]{"audit", root, "--manifest"}));
        AiwafConsole.main(new String[]{"init", "--source", root, "--output", manifest.toString()});
        assertEquals(0, AiwafConsole.handleAudit(new String[]{"audit", root, "--output", report.toString(), "--manifest", manifest.toString()}));
        assertTrue(java.nio.file.Files.readString(manifest).contains("injection_risks"));
    }

    @Test
    void init_can_read_routes_from_source(@org.junit.jupiter.api.io.TempDir java.nio.file.Path directory) throws Exception {
        java.nio.file.Files.writeString(directory.resolve("App.java"), VULNERABLE);
        java.nio.file.Path manifest = directory.resolve("out").resolve("paths.json");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try {
            AiwafConsole.main(new String[]{"init", "--source", directory.toString(), "--output", manifest.toString()});
        } finally {
            System.setOut(oldOut);
        }
        String content = java.nio.file.Files.readString(manifest);
        assertTrue(content.contains("/login"));
        assertTrue(content.contains("java-source"));
        // init saves the audit's findings for the route's checks.
        assertTrue(content.contains("injection_risks"));

        java.nio.file.Path plain = directory.resolve("plain.json");
        System.setOut(new PrintStream(new ByteArrayOutputStream()));
        try {
            AiwafConsole.main(new String[]{"init", "--source", directory.toString(), "--output", plain.toString(), "--no-audit"});
        } finally {
            System.setOut(oldOut);
        }
        assertTrue(!java.nio.file.Files.readString(plain).contains("injection_risks"));
    }

    @Test
    void geo_and_exempt_path_commands_are_supported() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try {
            AiwafConsole.main(new String[]{"geo", "add", "US"});
            AiwafConsole.main(new String[]{"geo", "list"});
            AiwafConsole.main(new String[]{"exempt-path", "add", "/health", "--reason", "Health check"});
            AiwafConsole.main(new String[]{"exempt-path", "list"});
        } finally {
            System.setOut(oldOut);
        }
        String output = out.toString();
        assertTrue(output.contains("US"));
        assertTrue(output.contains("/health"));
    }

    @Test
    void export_import_and_reset_commands_are_supported() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try {
            AiwafConsole.main(new String[]{"add", "blocklist", "198.51.100.99", "--reason", "test"});
            AiwafConsole.main(new String[]{"export", "aiwaf-console-export.bin"});
            AiwafConsole.main(new String[]{"reset"});
            AiwafConsole.main(new String[]{"import", "aiwaf-console-export.bin"});
            AiwafConsole.main(new String[]{"list", "blocklist"});
        } finally {
            System.setOut(oldOut);
        }
        String output = out.toString();
        assertTrue(output.contains("198.51.100.99"));
    }

    @Test
    void reset_flags_support_python_style_selective_behavior() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try {
            AiwafConsole.main(new String[]{"add", "blocklist", "198.51.100.130", "--reason", "test"});
            AiwafConsole.main(new String[]{"add", "allowlist", "203.0.113.130"});
            AiwafConsole.main(new String[]{"add", "keyword", "login"});

            AiwafConsole.main(new String[]{"reset", "--blocklist"});
            AiwafConsole.main(new String[]{"list", "blocklist"});
            AiwafConsole.main(new String[]{"list", "allowlist"});
            AiwafConsole.main(new String[]{"list", "keywords"});

            AiwafConsole.main(new String[]{"reset", "--exemptions-only"});
            AiwafConsole.main(new String[]{"list", "allowlist"});
            AiwafConsole.main(new String[]{"list", "keywords"});

            AiwafConsole.main(new String[]{"reset", "--keywords"});
            AiwafConsole.main(new String[]{"list", "keywords"});
        } finally {
            System.setOut(oldOut);
        }
        String output = out.toString();
        assertTrue(output.contains("203.0.113.130"));
        assertTrue(output.contains("login"));
    }

    @Test
    void import_command_returns_noop_for_invalid_file() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try {
            AiwafConsole.main(new String[]{"import", "missing-file.bin"});
        } finally {
            System.setOut(oldOut);
        }
        assertTrue(out.toString().contains("noop"));
    }

    @Test
    void list_all_and_path_exemptions_alias_are_supported() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try {
            AiwafConsole.main(new String[]{"add", "geo", "FR"});
            AiwafConsole.main(new String[]{"add", "path-exemptions", "/status", "--reason", "Status endpoint"});
            AiwafConsole.main(new String[]{"list", "all"});
            AiwafConsole.main(new String[]{"list", "path-exemptions"});
        } finally {
            System.setOut(oldOut);
        }
        String output = out.toString();
        assertTrue(output.contains("geo="));
        assertTrue(output.contains("path_exemptions="));
        assertTrue(output.contains("FR"));
        assertTrue(output.contains("/status"));
    }

    @Test
    void invalid_console_commands_print_usage() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try {
            AiwafConsole.main(new String[]{"list", "unknown-target"});
            AiwafConsole.main(new String[]{"geo", "unknown-action"});
        } finally {
            System.setOut(oldOut);
        }
        assertTrue(out.toString().contains("Usage:"));
    }

    @Test
    void invalid_add_remove_targets_return_noop() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try {
            AiwafConsole.main(new String[]{"add", "unknown", "value"});
            AiwafConsole.main(new String[]{"remove", "unknown", "value"});
        } finally {
            System.setOut(oldOut);
        }
        String output = out.toString();
        assertTrue(output.contains("noop"));
    }
}
