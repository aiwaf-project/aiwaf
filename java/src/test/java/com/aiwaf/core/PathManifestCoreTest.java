package com.aiwaf.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PathManifestCoreTest {
    @TempDir Path tempDir;

    @RestController
    @RequestMapping("/api/accounts")
    static class AccountController {
        @GetMapping("/{uuid}")
        ResponseEntity<String> account() { return ResponseEntity.ok("ok"); }

        @PostMapping("/login")
        ResponseEntity<String> login(@RequestBody Map<String, Object> body) { return ResponseEntity.ok("ok"); }
    }

    @Test
    void discovers_real_controller_paths_and_methods() {
        List<PathManifestCore.RouteInfo> routes = PathManifestCore.discoverControllerRoutes(AccountController.class);
        assertEquals(2, routes.size());
        assertTrue(routes.stream().anyMatch(route -> route.path().equals("/api/accounts/{uuid}")
                && route.httpMethods().equals(List.of("GET"))));
        assertTrue(routes.stream().anyMatch(route -> route.path().equals("/api/accounts/login")
                && route.httpMethods().equals(List.of("POST"))));
    }

    @Test
    void writes_shared_manifest_schema_and_stable_context_hash() throws Exception {
        List<PathManifestCore.RouteInfo> routes = PathManifestCore.discoverControllerRoutes(AccountController.class);
        Map<String, Object> first = PathManifestCore.buildManifest("spring", routes);
        Map<String, Object> second = PathManifestCore.buildManifest("spring", routes);
        assertEquals(first.get("context_hash"), second.get("context_hash"));

        Path output = tempDir.resolve(".aiwaf/paths.json");
        PathManifestCore.writeManifest(first, output);
        assertTrue(Files.isRegularFile(output));
        JsonNode manifest = new ObjectMapper().readTree(output.toFile());
        assertEquals("1.0", manifest.path("schema_version").asText());
        assertEquals("spring", manifest.path("framework").asText());
        assertTrue(manifest.path("routes").has("/api/accounts/{uuid}"));
        assertEquals("api", manifest.path("routes").path("/api/accounts/{uuid}").path("category").asText());
        assertEquals("GET", manifest.path("routes").path("/api/accounts/{uuid}").path("methods").get(0).asText());
    }

    @Test
    void excludes_internal_aiwaf_routes() throws Exception {
        java.lang.reflect.Method method = AccountController.class.getDeclaredMethod("account");
        Map<String, Object> manifest = PathManifestCore.buildManifest("spring", List.of(
                new PathManifestCore.RouteInfo("/aiwaf/status", List.of("GET"), AccountController.class, method)
        ));
        assertFalse(((Map<?, ?>) manifest.get("routes")).containsKey("/aiwaf/status"));
    }
}
