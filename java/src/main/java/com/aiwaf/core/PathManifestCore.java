package com.aiwaf.core;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PathManifestCore {
    private static final ObjectMapper mapper = new ObjectMapper();

    public static List<PathManifestEntry> generateManifest(List<RouteInfo> routes, String outputPath) {
        List<PathManifestEntry> entries = new ArrayList<>();

        for (RouteInfo route : routes) {
            ApiDetection api = ApiDetectionCore.detectApiEndpoint(route.method(), route.controllerClass(), route.path());
            Map<String, Object> auth = AuthDetectionCore.detectAuthEndpoint(route.path(), route.method(), route.controllerClass());
            entries.add(new PathManifestEntry(route.path(), route.httpMethods(), api, auth));
        }

        if (outputPath != null && !outputPath.isBlank()) {
            try {
                Map<String, Object> result = new HashMap<>();
                result.put("paths", entries);
                result.put("total", entries.size());

                mapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputPath), result);
                System.out.println("Manifest written to " + outputPath);
            } catch (IOException e) {
                System.err.println("Failed to write manifest to " + outputPath + ": " + e.getMessage());
            }
        }

        return entries;
    }

    public record RouteInfo(String path, List<String> httpMethods, Class<?> controllerClass, Method method) {}
}
