package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafEngine;
import com.aiwaf.runtime.RuntimeStorage;
import com.aiwaf.spring.annotations.AiwafExemptFrom;
import com.aiwaf.spring.annotations.AiwafOnly;
import com.aiwaf.spring.annotations.AiwafRequireProtection;
import org.junit.jupiter.api.Test;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.head;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class JavaFrameworkAiwafEndToEndTest {

    @RestController
    @RequestMapping("/")
    static class PathVarController {
        @GetMapping("items/{id}")
        @AiwafExemptFrom({"rate_limit"})
        public ResponseEntity<String> item(@PathVariable("id") String id) {
            return ResponseEntity.ok("item:" + id);
        }
    }

    @RestController
    @RequestMapping("/")
    @AiwafExemptFrom({"rate_limit"})
    static class ClassMethodAnnotationController {
        @GetMapping("ann-e2e/class-only")
        public ResponseEntity<String> classOnly() {
            return ResponseEntity.ok("ok");
        }

        @GetMapping("ann-e2e/method-require")
        @AiwafRequireProtection({"rate_limit"})
        public ResponseEntity<String> methodRequire() {
            return ResponseEntity.ok("ok");
        }

        @GetMapping("ann-e2e/method-only-header")
        @AiwafOnly({"header_validation"})
        public ResponseEntity<String> methodOnlyHeader() {
            return ResponseEntity.ok("ok");
        }
    }

    @Test
    void spring_rate_limit_per_path_and_global_modes() throws Exception {
        AiwafConfig perPath = new AiwafConfig();
        perPath.headerValidationEnabled = false;
        perPath.rateLimitMax = 1;
        perPath.rateLimitScope = AiwafConfig.RateLimitScope.PER_PATH;
        perPath.privateIpsExempted = false;
        MockMvc mvcPerPath = build(perPath);
        mvcPerPath.perform(get("/path-a")).andExpect(status().isOk());
        mvcPerPath.perform(get("/path-a")).andExpect(status().isTooManyRequests());
        mvcPerPath.perform(get("/path-b")).andExpect(status().isOk());

        AiwafConfig global = new AiwafConfig();
        global.headerValidationEnabled = false;
        global.rateLimitMax = 1;
        global.rateLimitScope = AiwafConfig.RateLimitScope.GLOBAL_IP;
        global.blockIpOnRateLimitBreach = true;
        global.privateIpsExempted = false;
        MockMvc mvcGlobal = build(global);
        mvcGlobal.perform(get("/path-a")).andExpect(status().isOk());
        mvcGlobal.perform(get("/path-a")).andExpect(status().isTooManyRequests());
        mvcGlobal.perform(get("/path-b")).andExpect(status().isForbidden());
    }

    @Test
    void spring_header_validation_rules() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.rateLimitEnabled = false;
        config.privateIpsExempted = false;
        MockMvc mvc = build(config);
        mvc.perform(get("/headers").header("User-Agent", "").header("Accept", "*/*")).andExpect(status().isForbidden());
        mvc.perform(get("/headers").header("User-Agent", "short").header("Accept", "*/*")).andExpect(status().isForbidden());
        mvc.perform(get("/headers")
                        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
                        .header("Accept", "text/html,application/xhtml+xml")
                        .header("Accept-Language", "en-US,en;q=0.9")
                        .header("Accept-Encoding", "gzip, deflate, br"))
                .andExpect(status().isOk());
    }

    @Test
    void spring_geo_block_allow_and_exemptions() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.headerValidationEnabled = false;
        config.rateLimitEnabled = false;
        config.geoBlockEnabled = true;
        config.geoBlockedCountries.add("US");
        config.geoAllowedCountries.add("US");
        config.geoExemptPaths.add("/exempted");
        config.privateIpsExempted = false;
        MockMvc mvc = build(config);

        mvc.perform(get("/blocked").header("X-Country", "US")).andExpect(status().isForbidden());
        mvc.perform(get("/exempted").header("X-Country", "US")).andExpect(status().isOk());
        mvc.perform(get("/blocked").header("X-Country", "CA")).andExpect(status().isForbidden());
    }

    @Test
    void spring_honeypot_uuid_and_malicious_path() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.headerValidationEnabled = false;
        config.rateLimitEnabled = false;
        config.honeypotEnabled = true;
        config.minFormTimeSeconds = 5.0;
        config.uuidTamperEnabled = true;
        config.ipKeywordBlockEnabled = true;
        config.privateIpsExempted = false;
        MockMvc mvc = build(config);

        mvc.perform(get("/form")).andExpect(status().isOk());
        mvc.perform(post("/form")).andExpect(status().isForbidden());
        mvc.perform(get("/uuid").queryParam("uuid", "invalid")).andExpect(status().isForbidden());
        mvc.perform(get("/uuid").queryParam("uuid", "550e8400-e29b-41d4-a716-446655440000")).andExpect(status().isOk());
        mvc.perform(get("/.env")).andExpect(status().isForbidden());
    }

    @Test
    void spring_path_rules_and_method_validation() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.privateIpsExempted = false;
        config.pathRules.add(new AiwafConfig.PathRule("/myapp/api/", true, 1000));
        config.pathRules.add(new AiwafConfig.PathRule("/myapp/", false, 1));
        config.methodValidationEnabled = true;
        config.allowedMethods.clear();
        config.allowedMethods.add("GET");
        MockMvc mvc = build(config);

        mvc.perform(get("/myapp/api/data").header("User-Agent", "").header("Accept", "*/*")).andExpect(status().isOk());
        mvc.perform(get("/myapp/ui").header("User-Agent", "").header("Accept", "*/*")).andExpect(status().isForbidden());
        mvc.perform(get("/myapp/ui").header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isOk());
        mvc.perform(get("/myapp/ui").header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isTooManyRequests());
        mvc.perform(post("/myapp/ui")).andExpect(status().isMethodNotAllowed());
    }

    @Test
    void spring_private_ip_exemption_can_be_disabled() throws Exception {
        AiwafConfig exempt = new AiwafConfig();
        exempt.rateLimitMax = 1;
        exempt.privateIpsExempted = true;
        exempt.headerValidationEnabled = false;
        MockMvc mvcExempt = build(exempt);
        mvcExempt.perform(get("/path-a")).andExpect(status().isOk());
        mvcExempt.perform(get("/path-a")).andExpect(status().isOk());

        AiwafConfig strict = new AiwafConfig();
        strict.rateLimitMax = 1;
        strict.privateIpsExempted = false;
        strict.headerValidationEnabled = false;
        MockMvc mvcStrict = build(strict);
        mvcStrict.perform(get("/path-a")).andExpect(status().isOk());
        mvcStrict.perform(get("/path-a")).andExpect(status().isTooManyRequests());
    }

    @Test
    void spring_auto_exempt_path_prefix_skips_controls() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.rateLimitMax = 1;
        config.privateIpsExempted = false;
        config.autoExemptPathPrefixes.add("/myapp/api/");
        MockMvc mvc = build(config);

        mvc.perform(get("/myapp/api/data")).andExpect(status().isOk());
        mvc.perform(get("/myapp/api/data")).andExpect(status().isOk());
    }

    @Test
    void spring_annotation_route_exemptions_match_python_logic() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.headerValidationEnabled = true;
        MockMvc mvc = build(config);

        mvc.perform(get("/ann/exempt")
                        .header("User-Agent", "curl/8.0.1")
                        .header("Accept", "*/*")
                        .header("X-Forwarded-For", "198.51.100.10"))
                .andExpect(status().isOk());
        mvc.perform(get("/ann/exempt")
                        .header("User-Agent", "curl/8.0.1")
                        .header("Accept", "*/*")
                        .header("X-Forwarded-For", "198.51.100.10"))
                .andExpect(status().isOk());

        mvc.perform(get("/ann/partial").header("X-Forwarded-For", "198.51.100.11")
                        .header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isOk());
        mvc.perform(get("/ann/partial").header("X-Forwarded-For", "198.51.100.11")
                        .header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isOk());

        mvc.perform(get("/ann/only").header("X-Forwarded-For", "198.51.100.12")
                        .header("User-Agent", "curl/8.0.1")
                        .header("Accept", "*/*"))
                .andExpect(status().isOk());
        mvc.perform(get("/ann/only").header("X-Forwarded-For", "198.51.100.12")
                        .header("User-Agent", "curl/8.0.1")
                        .header("Accept", "*/*"))
                .andExpect(status().isTooManyRequests());

        mvc.perform(get("/ann/required").header("X-Forwarded-For", "198.51.100.13")
                        .header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isOk());
        mvc.perform(get("/ann/required").header("X-Forwarded-For", "198.51.100.13")
                        .header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isTooManyRequests());
    }

    @Test
    void spring_forwarded_ip_is_used_for_rate_limit_identity() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.headerValidationEnabled = false;
        config.privateIpsExempted = true;
        MockMvc mvc = build(config);

        mvc.perform(get("/path-a").header("X-Forwarded-For", "198.51.100.21")).andExpect(status().isOk());
        mvc.perform(get("/path-a").header("X-Forwarded-For", "198.51.100.21")).andExpect(status().isTooManyRequests());
    }

    @Test
    void spring_forwarded_ip_uses_rightmost_untrusted_hop() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.headerValidationEnabled = false;
        config.privateIpsExempted = false;
        MockMvc mvc = build(config);

        String chain = "unknown, 198.51.100.41, 203.0.113.41";
        mvc.perform(get("/path-a").header("X-Forwarded-For", chain)).andExpect(status().isOk());
        mvc.perform(get("/path-a").header("X-Forwarded-For", "203.0.113.41")).andExpect(status().isTooManyRequests());
    }

    @Test
    void spring_real_ip_fallback_is_used_when_forwarded_for_is_unknown() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.headerValidationEnabled = false;
        config.privateIpsExempted = false;
        MockMvc mvc = build(config);

        mvc.perform(get("/path-a")
                        .header("X-Forwarded-For", "unknown")
                        .header("X-Real-IP", "198.51.100.42"))
                .andExpect(status().isOk());
        mvc.perform(get("/path-a")
                        .header("X-Forwarded-For", "unknown")
                        .header("X-Real-IP", "198.51.100.42"))
                .andExpect(status().isTooManyRequests());
    }

    @Test
    void spring_can_disable_middlewares_globally() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.headerValidationEnabled = true;
        config.disabledMiddlewares.add("rate_limit");
        config.disabledMiddlewares.add("header_validation");
        MockMvc mvc = build(config);

        mvc.perform(get("/path-a").header("X-Forwarded-For", "198.51.100.31")
                        .header("User-Agent", "curl/8.0.1")
                        .header("Accept", "*/*"))
                .andExpect(status().isOk());
        mvc.perform(get("/path-a").header("X-Forwarded-For", "198.51.100.31")
                        .header("User-Agent", "curl/8.0.1")
                        .header("Accept", "*/*"))
                .andExpect(status().isOk());
    }

    @Test
    void spring_can_enable_only_selected_middlewares() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.headerValidationEnabled = true;
        config.enabledMiddlewares.add("rate_limit");
        MockMvc mvc = build(config);

        mvc.perform(get("/path-a").header("X-Forwarded-For", "198.51.100.32")
                        .header("User-Agent", "curl/8.0.1")
                        .header("Accept", "*/*"))
                .andExpect(status().isOk());
        mvc.perform(get("/path-a").header("X-Forwarded-For", "198.51.100.32")
                        .header("User-Agent", "curl/8.0.1")
                        .header("Accept", "*/*"))
                .andExpect(status().isTooManyRequests());
    }

    @Test
    void spring_geo_block_uses_runtime_geo_store() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        config.geoBlockEnabled = true;
        config.headerValidationEnabled = false;
        config.rateLimitEnabled = false;
        MockMvc mvc = build(config);

        RuntimeStorage.getGeoBlockStore().addCountry("US");
        mvc.perform(get("/blocked")
                        .header("X-Country", "US")
                        .header("X-Forwarded-For", "198.51.100.33"))
                .andExpect(status().isForbidden());
    }

    @Test
    void spring_head_can_override_required_headers_by_method() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        config.rateLimitEnabled = false;
        config.requiredHeadersByMethod.put("HEAD", java.util.List.of());
        MockMvc mvc = build(config);

        mvc.perform(head("/headers")
                        .header("User-Agent", "EmailScanner/1.0")
                        .header("X-Forwarded-For", "198.51.100.34"))
                .andExpect(status().isOk());
    }

    @Test
    void spring_path_variable_routes_apply_annotation_route_decisions() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.headerValidationEnabled = false;

        PathVarController controller = new PathVarController();
        MockMvc mvc = MockMvcBuilders.standaloneSetup(controller)
                .addFilters(new AiwafFilter(new AiwafEngine(config), controller))
                .build();

        mvc.perform(get("/items/1").header("X-Forwarded-For", "198.51.100.50")).andExpect(status().isOk());
        mvc.perform(get("/items/1").header("X-Forwarded-For", "198.51.100.50")).andExpect(status().isOk());
    }

    @Test
    void spring_class_and_method_annotation_interactions_apply_in_e2e() throws Exception {
        AiwafConfig config = new AiwafConfig();
        config.privateIpsExempted = false;
        config.rateLimitEnabled = true;
        config.rateLimitMax = 1;
        config.headerValidationEnabled = true;

        ClassMethodAnnotationController controller = new ClassMethodAnnotationController();
        MockMvc mvc = MockMvcBuilders.standaloneSetup(controller)
                .addFilters(new AiwafFilter(new AiwafEngine(config), controller))
                .build();

        mvc.perform(get("/ann-e2e/class-only")
                        .header("X-Forwarded-For", "198.51.100.60")
                        .header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isOk());
        mvc.perform(get("/ann-e2e/class-only")
                        .header("X-Forwarded-For", "198.51.100.60")
                        .header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isOk());

        mvc.perform(get("/ann-e2e/method-require")
                        .header("X-Forwarded-For", "198.51.100.61")
                        .header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isOk());
        mvc.perform(get("/ann-e2e/method-require")
                        .header("X-Forwarded-For", "198.51.100.61")
                        .header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isTooManyRequests());

        mvc.perform(get("/ann-e2e/method-only-header")
                        .header("X-Forwarded-For", "198.51.100.62")
                        .header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isOk());
        mvc.perform(get("/ann-e2e/method-only-header")
                        .header("X-Forwarded-For", "198.51.100.62")
                        .header("User-Agent", "Mozilla/5.0 Test Browser")
                        .header("Accept", "text/html")
                        .header("Accept-Language", "en-US")
                        .header("Accept-Encoding", "gzip"))
                .andExpect(status().isOk());
    }

    private static MockMvc build(AiwafConfig config) {
        TestController controller = new TestController();
        return MockMvcBuilders.standaloneSetup(controller)
                .addFilters(new AiwafFilter(new AiwafEngine(config), controller))
                .build();
    }
}
