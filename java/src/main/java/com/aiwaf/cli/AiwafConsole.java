package com.aiwaf.cli;

import com.aiwaf.core.WhoisLookup;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class AiwafConsole {
    private AiwafConsole() {}

    public static void main(String[] args) {
        if (args == null || args.length == 0) {
            printUsage();
            return;
        }
        AiwafManager manager = new AiwafManager();
        String cmd = args[0].toLowerCase(Locale.ROOT);
        try {
            switch (cmd) {
                case "stats" -> System.out.println(manager.stats());
                case "list" -> handleList(manager, args);
                case "add" -> handleAdd(manager, args);
                case "remove" -> handleRemove(manager, args);
                case "export" -> handleExport(manager, args);
                case "import" -> handleImport(manager, args);
                case "reset" -> handleReset(manager, args);
                case "geo" -> handleGeo(manager, args);
                case "exempt-path", "path-exemptions" -> handleExemptPath(manager, args);
                case "whois" -> handleWhois(args);
                case "generate-manifest" -> handleGenerateManifest(args);
                default -> printUsage();
            }
        } catch (Exception ex) {
            System.err.println("Error: " + ex.getMessage());
            System.exit(1);
        }
    }

    private static void handleList(AiwafManager manager, String[] args) {
        String target = args.length > 1 ? args[1].toLowerCase(Locale.ROOT) : "all";
        switch (target) {
            case "whitelist" -> System.out.println(manager.listWhitelistIps());
            case "blacklist" -> System.out.println(manager.listBlacklistIps());
            case "keywords" -> System.out.println(manager.listKeywords(100));
            case "geo" -> System.out.println(manager.listGeoBlockedCountries());
            case "exempt-path", "path-exemptions" -> System.out.println(manager.listPathExemptions());
            case "all" -> {
                System.out.println("whitelist=" + manager.listWhitelistIps());
                System.out.println("blacklist=" + manager.listBlacklistIps());
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
            case "whitelist" -> manager.addWhitelistIp(value, reason);
            case "blacklist" -> manager.addBlacklistIp(value, reason);
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
            case "whitelist" -> manager.removeWhitelistIp(value);
            case "blacklist" -> manager.removeBlacklistIp(value);
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
                "  aiwaf-cli list [all|whitelist|blacklist|keywords|geo|exempt-path]",
                "  aiwaf-cli add <whitelist|blacklist|keyword|geo|exempt-path> <value> [--reason <text>]",
                "  aiwaf-cli remove <whitelist|blacklist|keyword|geo|exempt-path> <value>",
                "  aiwaf-cli export <file>",
                "  aiwaf-cli import <file>",
                "  aiwaf-cli reset [--blacklist|--exemptions|--keywords|--blacklist-only|--exemptions-only]",
                "  aiwaf-cli geo <list|add|remove> [country]",
                "  aiwaf-cli exempt-path <list|add|remove> [path] [--reason text]",
                "  aiwaf-cli whois <target>",
                "  aiwaf-cli generate-manifest <output.json> --class <ControllerClassName>"
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

    private static void handleGenerateManifest(String[] args) {
        String output = args.length > 1 && !args[1].startsWith("--") ? args[1] : "manifest.json";
        String clsName = parseOption(args, "--class");
        if (clsName == null) {
            System.out.println("Usage: aiwaf-cli generate-manifest <output.json> --class <ControllerClassName>");
            return;
        }
        try {
            Class<?> cls = Class.forName(clsName);
            List<com.aiwaf.core.PathManifestCore.RouteInfo> routes = new java.util.ArrayList<>();
            for (java.lang.reflect.Method m : cls.getMethods()) {
                if (m.getDeclaringClass() == Object.class) continue;
                routes.add(new com.aiwaf.core.PathManifestCore.RouteInfo("/" + m.getName(), List.of("GET", "POST"), cls, m));
            }
            com.aiwaf.core.PathManifestCore.generateManifest(routes, output);
        } catch (Exception e) {
            System.err.println("Error generating manifest: " + e.getMessage());
        }
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
        boolean blacklistOnly = hasArg(args, "--blacklist-only");
        boolean exemptionsOnly = hasArg(args, "--exemptions-only");
        if (blacklistOnly) {
            System.out.println(manager.resetSelective(true, false, false) ? "ok" : "noop");
            return;
        }
        if (exemptionsOnly) {
            System.out.println(manager.resetSelective(false, true, false) ? "ok" : "noop");
            return;
        }

        boolean blacklist = hasArg(args, "--blacklist");
        boolean exemptions = hasArg(args, "--exemptions");
        boolean keywords = hasArg(args, "--keywords");
        if (blacklist || exemptions || keywords) {
            System.out.println(manager.resetSelective(blacklist, exemptions, keywords) ? "ok" : "noop");
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
