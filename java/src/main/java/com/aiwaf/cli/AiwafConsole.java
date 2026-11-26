package com.aiwaf.cli;

import com.aiwaf.core.WhoisLookup;
import com.aiwaf.core.CoreCli;
import com.aiwaf.core.ModelArtifactIoCore;
import com.aiwaf.core.TrainedModelCore;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.nio.file.Files;
import java.nio.file.Path;

public final class AiwafConsole {
    private AiwafConsole() {}

    public static void main(String[] args) {
        if (args == null || args.length == 0) {
            printUsage();
            return;
        }
        String cmd = args[0].toLowerCase(Locale.ROOT);
        args = args.clone();
        if (args.length > 1 && java.util.Set.of("list", "add", "remove").contains(cmd)) {
            if (args[1].equals("blacklist")) args[1] = "blocklist";
            if (args[1].equals("whitelist")) args[1] = "allowlist";
        }
        if (cmd.equals("reset")) {
            for (int i = 1; i < args.length; i++) args[i] = args[i].replace("--blacklist", "--blocklist");
        }
        try {
            switch (cmd) {
                case "stats", "status" -> System.out.println(manager().stats());
                case "diagnose" -> handleDiagnose(manager(), args);
                case "list" -> handleList(manager(), args);
                case "blocked" -> System.out.println(manager().listBlocklist());
                case "unblock" -> handleUnblock(manager(), args);
                case "clear" -> System.out.println(manager().resetSelective(true, false, false) ? "ok" : "noop");
                case "add" -> handleAdd(manager(), args);
                case "remove" -> handleRemove(manager(), args);
                case "export" -> handleExport(manager(), args);
                case "import" -> handleImport(manager(), args);
                case "reset" -> handleReset(manager(), args);
                case "geo" -> handleGeo(manager(), args);
                case "exempt-path", "path-exemptions" -> handleExemptPath(manager(), args);
                case "whois" -> handleWhois(args);
                case "generate-manifest", "init" -> {
                    int status = handleGenerateManifest(args);
                    // A failure must fail the command; success returns, as the other commands do.
                    if (status != 0) System.exit(status);
                }
                case "audit" -> System.exit(handleAudit(args));
                case "train", "train-model" -> handleCoreCommand("train-model", args);
                case "replay" -> handleCoreCommand("replay", args);
                case "model" -> handleModel(args);
                case "logs" -> handleLogs(args);
                default -> printUsage();
            }
        } catch (Exception ex) {
            System.err.println("Error: " + ex.getMessage());
            System.exit(1);
        }
    }

    private static AiwafManager manager() {
        return new AiwafManager();
    }

    private static void handleList(AiwafManager manager, String[] args) {
        String target = args.length > 1 ? args[1].toLowerCase(Locale.ROOT) : "all";
        switch (target) {
            case "allowlist" -> System.out.println(manager.listAllowlistIps());
            case "blocklist" -> System.out.println(manager.listBlocklistIps());
            case "keywords" -> System.out.println(manager.listKeywords(100));
            case "geo" -> System.out.println(manager.listGeoBlockedCountries());
            case "exempt-path", "path-exemptions" -> System.out.println(manager.listPathExemptions());
            case "all" -> {
                System.out.println("allowlist=" + manager.listAllowlistIps());
                System.out.println("blocklist=" + manager.listBlocklistIps());
                System.out.println("keywords=" + manager.listKeywords(100));
                System.out.println("geo=" + manager.listGeoBlockedCountries());
                System.out.println("path_exemptions=" + manager.listPathExemptions());
            }
            default -> printUsage();
        }
    }

    private static void handleAdd(AiwafManager manager, String[] args) {
        if (args.length < 3) {
            printUsage();
            return;
        }
        String target = args[1].toLowerCase(Locale.ROOT);
        String value = args[2];
        String reason = parseOption(args, "--reason");
        boolean result = switch (target) {
            case "allowlist" -> manager.addAllowlistIp(value, reason);
            case "blocklist" -> manager.addBlocklistIp(value, reason);
            case "keyword" -> manager.addKeyword(value);
            case "geo" -> manager.addGeoBlockedCountry(value);
            case "exempt-path", "path-exemptions" -> manager.addPathExemption(value, reason);
            default -> false;
        };
        System.out.println(result ? "ok" : "noop");
    }

    private static void handleRemove(AiwafManager manager, String[] args) {
        if (args.length < 3) {
            printUsage();
            return;
        }
        String target = args[1].toLowerCase(Locale.ROOT);
        String value = args[2];
        boolean result = switch (target) {
            case "allowlist" -> manager.removeAllowlistIp(value);
            case "blocklist" -> manager.removeBlocklistIp(value);
            case "keyword" -> manager.removeKeyword(value);
            case "geo" -> manager.removeGeoBlockedCountry(value);
            case "exempt-path", "path-exemptions" -> manager.removePathExemption(value);
            default -> false;
        };
        System.out.println(result ? "ok" : "noop");
    }

    private static String parseOption(String[] args, String option) {
        for (int i = 0; i < args.length; i++) {
            if (option.equals(args[i]) && i + 1 < args.length) {
                return args[i + 1];
            }
        }
        return null;
    }

    private static void printUsage() {
        List<String> lines = List.of(
                "Usage:",
                "  aiwaf-cli stats",
                "  aiwaf-cli diagnose [--model <model.json>]",
                "  aiwaf-cli blocked | unblock <ip> | clear",
                "  aiwaf-cli list [all|allowlist|blocklist|keywords|geo|exempt-path]",
                "  aiwaf-cli add <allowlist|blocklist|keyword|geo|exempt-path> <value> [--reason <text>]",
                "  aiwaf-cli remove <allowlist|blocklist|keyword|geo|exempt-path> <value>",
                "  aiwaf-cli export <file>",
                "  aiwaf-cli import <file>",
                "  aiwaf-cli reset [--blocklist|--exemptions|--keywords|--blocklist-only|--exemptions-only]",
                "  aiwaf-cli geo <list|add|remove> [country]",
                "  aiwaf-cli exempt-path <list|add|remove> [path] [--reason text]",
                "  aiwaf-cli whois <target>",
                "  aiwaf-cli train --events <events.json> --out <model.json> [--backend java|fastr]",
                "  aiwaf-cli replay --cases <cases.json>",
                "  aiwaf-cli model status [--path <model.json>]",
                "  aiwaf-cli logs [--dir aiwaf_logs]",
                "  aiwaf-cli init --app <SpringApplicationClass> [--output .aiwaf/paths.json] [--audit-source <folder>] [--no-audit]",
                "  aiwaf-cli init --source <folder> [--output .aiwaf/paths.json] [--no-audit]",
                "  aiwaf-cli audit [path] [--format text|json] [--output file] [--include-tests] [--fail-on-findings] [--manifest file]",
                "  aiwaf-cli generate-manifest [output.json] --class <ControllerClassName>"
        );
        for (String line : lines) {
            System.out.println(line);
        }
    }

    private static void handleGeo(AiwafManager manager, String[] args) {
        if (args.length < 2) {
            printUsage();
            return;
        }
        String action = args[1].toLowerCase(Locale.ROOT);
        switch (action) {
            case "list" -> System.out.println(manager.listGeoBlockedCountries());
            case "add" -> System.out.println((args.length > 2 && manager.addGeoBlockedCountry(args[2])) ? "ok" : "noop");
            case "remove" -> System.out.println((args.length > 2 && manager.removeGeoBlockedCountry(args[2])) ? "ok" : "noop");
            default -> printUsage();
        }
    }

    private static void handleExemptPath(AiwafManager manager, String[] args) {
        if (args.length < 2) {
            printUsage();
            return;
        }
        String action = args[1].toLowerCase(Locale.ROOT);
        String reason = parseOption(args, "--reason");
        switch (action) {
            case "list" -> System.out.println(manager.listPathExemptions());
            case "add" -> System.out.println((args.length > 2 && manager.addPathExemption(args[2], reason)) ? "ok" : "noop");
            case "remove" -> System.out.println((args.length > 2 && manager.removePathExemption(args[2])) ? "ok" : "noop");
            default -> printUsage();
        }
    }

    private static void handleWhois(String[] args) {
        if (args.length < 2) {
            printUsage();
            return;
        }
        String target = args[1];
        try {
            Map<String, String> result = WhoisLookup.runWhoisLookup(target);
            System.out.println("WHOIS result: " + result);
        } catch (IOException ex) {
            System.out.println("python-whois is not installed or whois command unavailable");
        }
    }

    private static void handleUnblock(AiwafManager manager, String[] args) {
        System.out.println(args.length > 1 && manager.removeBlocklistIp(args[1]) ? "ok" : "noop");
    }

    private static void handleCoreCommand(String command, String[] args) {
        String[] delegated = new String[args.length];
        delegated[0] = command;
        if (args.length > 1) System.arraycopy(args, 1, delegated, 1, args.length - 1);
        int status = CoreCli.main(delegated);
        System.out.println(status == 0 ? "ok" : "error (exit " + status + ")");
    }

    private static void handleDiagnose(AiwafManager manager, String[] args) {
        String modelPath = parseOption(args, "--model");
        if (modelPath == null) modelPath = "model.json";
        Package pkg = AiwafConsole.class.getPackage();
        String version = pkg == null || pkg.getImplementationVersion() == null
                ? "development" : pkg.getImplementationVersion();
        System.out.println("AIWAF Java diagnostics");
        System.out.println("version=" + version);
        System.out.println("java=" + System.getProperty("java.runtime.version", "unknown"));
        System.out.println("storage=" + manager.stats());
        System.out.println("model_path=" + Path.of(modelPath).toAbsolutePath().normalize());
        System.out.println("model=" + (ModelArtifactIoCore.load(modelPath) == null ? "unavailable" : "ready"));
    }

    private static void handleModel(String[] args) {
        String action = args.length > 1 ? args[1].toLowerCase(Locale.ROOT) : "status";
        if (!"status".equals(action)) {
            printUsage();
            return;
        }
        String path = parseOption(args, "--path");
        if (path == null) path = "model.json";
        TrainedModelCore model = ModelArtifactIoCore.load(path);
        if (model == null) {
            System.out.println("model=unavailable path=" + Path.of(path).toAbsolutePath().normalize());
            return;
        }
        System.out.println("model=ready type=" + model.modelType() + " version=" + model.version());
    }

    private static void handleLogs(String[] args) throws IOException {
        String configured = parseOption(args, "--dir");
        Path directory = Path.of(configured == null ? "aiwaf_logs" : configured);
        if (!Files.isDirectory(directory)) {
            System.out.println("logs=unavailable dir=" + directory.toAbsolutePath().normalize());
            return;
        }
        try (var files = Files.list(directory)) {
            files.filter(Files::isRegularFile)
                    .sorted((left, right) -> Long.compare(lastModified(right), lastModified(left)))
                    .forEach(path -> System.out.println(path + " bytes=" + size(path)));
        }
    }

    private static long lastModified(Path path) {
        try { return Files.getLastModifiedTime(path).toMillis(); } catch (IOException ignored) { return 0L; }
    }

    private static long size(Path path) {
        try { return Files.size(path); } catch (IOException ignored) { return 0L; }
    }

    /**
     * init: writes .aiwaf/paths.json from the app's routes. Returns the exit status: 0 when it
     * wrote a manifest, 1 when it couldn't, or found no routes, and 2 for usage errors. A manifest
     * with no routes is never written: it would look configured while protecting no route, and
     * would replace one that works.
     */
    static int handleGenerateManifest(String[] args) {
        String outputOption = parseOption(args, "--output");
        String output = outputOption != null
                ? outputOption
                : (args.length > 1 && !args[1].startsWith("--")
                        ? args[1] : com.aiwaf.core.PathManifestCore.DEFAULT_MANIFEST_PATH);
        String sourceRoot = parseOption(args, "--source");
        String appName = parseOption(args, "--app");
        String clsName = parseOption(args, "--class");
        if (sourceRoot == null && appName == null && clsName == null) {
            System.err.println("Usage: aiwaf-cli init --app <SpringApplicationClass> [--output .aiwaf/paths.json]");
            System.err.println("   or: aiwaf-cli init --source <folder> [--output .aiwaf/paths.json]");
            System.err.println("   or: aiwaf-cli init --class <ControllerClass> [--output .aiwaf/paths.json]");
            return 2;
        }
        try {
            if (sourceRoot != null) {
                com.aiwaf.core.JavaSourceAnalysisCore.Report report =
                        com.aiwaf.core.JavaSourceAnalysisCore.analyze(java.nio.file.Path.of(sourceRoot), false);
                report.errors().forEach(error -> System.err.println(error.get("file") + ": not analyzed: " + error.get("error")));
                Map<String, Object> manifest = new java.util.LinkedHashMap<>(
                        com.aiwaf.core.PathManifestCore.buildManifestFromSource("java-source", report.routes()));
                if (!hasFlag(args, "--no-audit")) {
                    // The same parse found the routes and the injection findings.
                    com.aiwaf.core.RouteRiskCore.applyReport(report, manifest, inputSources(List.of()));
                } else {
                    // The parse read the inputs; without its findings saved, they aren't a complete list.
                    com.aiwaf.core.JavaSourceAnalysisCore.Report inputsOnly = new com.aiwaf.core.JavaSourceAnalysisCore.Report(
                            report.filesScanned(), List.of(), report.errors(), report.routes(), 0, 0, report.inputs());
                    com.aiwaf.core.RouteRiskCore.applyReport(inputsOnly, manifest, inputSources(List.of()));
                    manifest.remove("injection_audit");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> routes = (Map<String, Object>) manifest.get("routes");
                    com.aiwaf.core.InputSourcesCore.markNotAudited(routes);
                }
                writeFound(manifest, output, sourceRoot + " (java-source)");
                printManifestSummary(output, manifest);
                return 0;
            }
            List<com.aiwaf.core.PathManifestCore.RouteInfo> routes;
            String where;
            if (appName != null) {
                routes = com.aiwaf.spring.SpringPathManifest.launchAndDiscover(appName);
                where = appName + " (spring)";
            } else {
                Class<?> cls;
                try {
                    cls = Class.forName(clsName);
                } catch (ClassNotFoundException ex) {
                    throw new IllegalStateException("Controller class not found: " + clsName
                            + " (check the name and that the app's classes are on the classpath)", ex);
                }
                routes = com.aiwaf.core.PathManifestCore.discoverControllerRoutes(cls);
                where = clsName + " (spring)";
            }
            Map<String, Object> manifest = com.aiwaf.core.PathManifestCore.buildManifest("spring", routes);
            if (hasFlag(args, "--no-audit")) {
                // The framework's and declared inputs, without the audit: types only.
                @SuppressWarnings("unchecked")
                Map<String, Object> found = (Map<String, Object>) manifest.get("routes");
                com.aiwaf.core.InputSourcesCore.finalizeWithoutCode(found, inputSources(routes));
            }
            writeFound(manifest, output, where);
            printManifestSummary(output, manifest);
            auditIntoManifest(args, output, routes);
            return 0;
        } catch (LinkageError ex) {
            System.err.println("Error generating manifest: source analysis needs a JDK, and this Java runtime has no compiler.");
            return 1;
        } catch (IOException | RuntimeException ex) {
            System.err.println("Error generating manifest: " + ex.getMessage());
            return 1;
        }
    }

    /** Writes init's manifest, or throws when it has no routes, leaving any manifest there as it was. */
    static void writeFound(Map<String, Object> manifest, String output, String where) {
        Object routes = manifest.get("routes");
        if (!(routes instanceof Map<?, ?> map) || map.isEmpty()) {
            throw new IllegalStateException("No routes found in " + where + ", so " + output + " was not written. "
                    + "Check that the app's controllers are loaded (or that --source is the folder holding them).");
        }
        com.aiwaf.core.PathManifestCore.writeManifest(manifest, java.nio.file.Path.of(output));
    }

    private static boolean hasFlag(String[] args, String flag) {
        return java.util.Arrays.asList(args).contains(flag);
    }

    /**
     * What init combines with the source scan: .aiwaf/inputs.json, AIWAF_OPENAPI_SPEC and recorded
     * inputs, plus the framework's and @AiwafInputs declarations for the given live routes.
     */
    static com.aiwaf.core.InputSourcesCore.Sources inputSources(java.util.Collection<com.aiwaf.core.PathManifestCore.RouteInfo> routes) {
        com.aiwaf.core.InputSourcesCore.Sources base;
        try {
            base = com.aiwaf.core.InputSourcesCore.sourcesFromEnvironment(java.nio.file.Path.of("."), System.getenv());
        } catch (IOException | RuntimeException ex) {
            System.err.println("Declared, OpenAPI or recorded inputs skipped: " + ex.getMessage());
            base = com.aiwaf.core.InputSourcesCore.Sources.none();
        }
        return com.aiwaf.core.InputSourcesCore.forRoutes(routes, base);
    }

    /** Saves injection findings from source into a written manifest. A failed scan leaves it unchanged. */
    static void auditIntoManifest(String[] args, String output) {
        auditIntoManifest(args, output, List.of());
    }

    static void auditIntoManifest(String[] args, String output, java.util.Collection<com.aiwaf.core.PathManifestCore.RouteInfo> routes) {
        if (hasFlag(args, "--no-audit")) {
            return;
        }
        String source = parseOption(args, "--audit-source");
        if (source == null) {
            source = java.nio.file.Files.isDirectory(java.nio.file.Path.of("src", "main", "java")) ? "src/main/java" : ".";
        }
        try {
            com.aiwaf.core.JavaSourceAnalysisCore.Report report =
                    com.aiwaf.core.JavaSourceAnalysisCore.analyze(java.nio.file.Path.of(source), false);
            System.out.println("Injection checks: " + com.aiwaf.core.RouteRiskCore.formatCounts(
                    com.aiwaf.core.RouteRiskCore.updateManifest(report, java.nio.file.Path.of(output), inputSources(routes)), output));
        } catch (IOException | RuntimeException ex) {
            System.err.println("Injection audit skipped: " + ex.getMessage());
        } catch (LinkageError ex) {
            System.err.println("Injection audit skipped: source analysis needs a JDK, and this Java runtime has no compiler.");
        }
    }

    /** Reports possible injection paths. Returns the exit status. */
    static int handleAudit(String[] args) {
        String format = "text";
        String output = null;
        String path = ".";
        boolean includeTests = false;
        boolean failOnFindings = false;
        String manifest = null;
        String baseline = null;
        for (int index = 1; index < args.length; index++) {
            switch (args[index]) {
                case "--baseline" -> baseline = index + 1 < args.length ? args[++index] : "";
                case "--format" -> format = index + 1 < args.length ? args[++index] : "";
                case "--output" -> output = index + 1 < args.length ? args[++index] : "";
                case "--include-tests" -> includeTests = true;
                case "--fail-on-findings" -> failOnFindings = true;
                case "--manifest" -> manifest = index + 1 < args.length ? args[++index] : "";
                default -> path = args[index];
            }
        }
        if (!List.of("text", "json", "sarif").contains(format) || "".equals(output) || "".equals(manifest) || "".equals(baseline)) {
            System.err.println("Usage: aiwaf-cli audit [path] [--format text|json|sarif] [--output file] [--include-tests] "
                    + "[--fail-on-findings] [--manifest file] [--baseline file]");
            return 2;
        }
        try {
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            com.aiwaf.core.JavaSourceAnalysisCore.Report report =
                    com.aiwaf.core.JavaSourceAnalysisCore.analyze(java.nio.file.Path.of(path), includeTests);
            if (baseline != null) {
                @SuppressWarnings("unchecked")
                Map<String, Object> earlier = mapper.readValue(java.nio.file.Path.of(baseline).toFile(), Map.class);
                report = com.aiwaf.core.JavaSourceAnalysisCore.applyBaseline(report, earlier);
            }
            String content = switch (format) {
                case "json" -> mapper.writerWithDefaultPrettyPrinter().writeValueAsString(report.toMap());
                case "sarif" -> mapper.writerWithDefaultPrettyPrinter().writeValueAsString(
                        com.aiwaf.core.JavaSourceAnalysisCore.toSarif(report));
                default -> report.toText();
            };
            if (output != null) {
                java.nio.file.Files.writeString(java.nio.file.Path.of(output), content + System.lineSeparator());
            } else {
                System.out.println(content);
            }
            if (manifest != null) {
                System.out.println(com.aiwaf.core.RouteRiskCore.formatCounts(
                        com.aiwaf.core.RouteRiskCore.updateManifest(report, java.nio.file.Path.of(manifest)), manifest));
            }
            return failOnFindings && !report.findings().isEmpty() ? 1 : 0;
        } catch (IOException | IllegalStateException | IllegalArgumentException ex) {
            System.err.println("Error: " + ex.getMessage());
            return 2;
        } catch (LinkageError ex) {
            System.err.println("Error: source analysis needs a JDK, and this Java runtime has no compiler.");
            return 2;
        }
    }

    private static void printManifestSummary(String output, Map<String, Object> manifest) {
        Object routes = manifest.get("routes");
        int count = routes instanceof Map<?, ?> map ? map.size() : 0;
        System.out.println("Generated " + output);
        System.out.println("Framework: " + manifest.getOrDefault("framework", "spring"));
        System.out.println("Routes: " + count);
        System.out.println("Context hash: " + manifest.getOrDefault("context_hash", ""));
    }

    private static void handleExport(AiwafManager manager, String[] args) {
        if (args.length < 2) {
            printUsage();
            return;
        }
        System.out.println(manager.exportConfig(args[1]) ? "ok" : "noop");
    }

    private static void handleImport(AiwafManager manager, String[] args) {
        if (args.length < 2) {
            printUsage();
            return;
        }
        System.out.println(manager.importConfig(args[1]) ? "ok" : "noop");
    }

    private static void handleReset(AiwafManager manager, String[] args) {
        boolean blocklistOnly = hasArg(args, "--blocklist-only");
        boolean exemptionsOnly = hasArg(args, "--exemptions-only");
        if (blocklistOnly) {
            System.out.println(manager.resetSelective(true, false, false) ? "ok" : "noop");
            return;
        }
        if (exemptionsOnly) {
            System.out.println(manager.resetSelective(false, true, false) ? "ok" : "noop");
            return;
        }

        boolean blocklist = hasArg(args, "--blocklist");
        boolean exemptions = hasArg(args, "--exemptions");
        boolean keywords = hasArg(args, "--keywords");
        if (blocklist || exemptions || keywords) {
            System.out.println(manager.resetSelective(blocklist, exemptions, keywords) ? "ok" : "noop");
            return;
        }
        System.out.println(manager.resetAll() ? "ok" : "noop");
    }

    private static boolean hasArg(String[] args, String value) {
        for (String arg : args) {
            if (value.equals(arg)) {
                return true;
            }
        }
        return false;
    }
}
