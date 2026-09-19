package com.aiwaf.cli;

import org.junit.jupiter.api.Test;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RouteShellHelpersTest {

    @RestController
    @RequestMapping("/")
    static class DemoController {
        @GetMapping("api/v1/users")
        public ResponseEntity<String> users() {
            return ResponseEntity.ok("ok");
        }

        @GetMapping("health")
        public ResponseEntity<String> health() {
            return ResponseEntity.ok("ok");
        }
    }

    @Test
    void route_tree_building_and_resolve_target_work() {
        List<String> routes = RouteShellHelpers.collectRoutes(new DemoController());
        RouteNode root = RouteShellHelpers.buildTree(routes);

        RouteNode api = RouteShellHelpers.resolveTarget(root, "api");
        assertNotNull(api);
        RouteNode v1 = RouteShellHelpers.resolveTarget(api, "v1");
        assertNotNull(v1);
        RouteNode users = RouteShellHelpers.resolveTarget(v1, "users");
        assertNotNull(users);
        assertTrue(users.endpoint());

        RouteNode health = RouteShellHelpers.resolveTarget(root, "health");
        assertNotNull(health);
        assertTrue(health.endpoint());
    }

    @Test
    void exempt_flow_adds_current_path() {
        AiwafManager manager = new AiwafManager("aiwaf-route-shell-test.bin");
        List<String> routes = RouteShellHelpers.collectRoutes(new DemoController());
        RouteNode root = RouteShellHelpers.buildTree(routes);
        RouteNode api = RouteShellHelpers.resolveTarget(root, "api");
        assertNotNull(api);

        assertTrue(RouteShellHelpers.exemptCurrentPath(manager, api, "Polling endpoint"));
        assertTrue(manager.listPathExemptions().contains("/api/"));
        assertEquals("/api/", RouteShellHelpers.routePath(api));
    }

    @Test
    void resolve_target_supports_index_parent_and_unknown_tokens() {
        List<String> routes = RouteShellHelpers.collectRoutes(new DemoController());
        RouteNode root = RouteShellHelpers.buildTree(routes);

        RouteNode api = RouteShellHelpers.resolveTarget(root, "api");
        assertNotNull(api);
        assertEquals(root, RouteShellHelpers.resolveTarget(api, ".."));

        RouteNode first = RouteShellHelpers.resolveTarget(root, "1");
        assertNotNull(first);

        RouteNode unchanged = RouteShellHelpers.resolveTarget(api, "does-not-exist");
        assertEquals(api, unchanged);

        RouteNode same = RouteShellHelpers.resolveTarget(api, ".");
        assertEquals(api, same);
    }
}
