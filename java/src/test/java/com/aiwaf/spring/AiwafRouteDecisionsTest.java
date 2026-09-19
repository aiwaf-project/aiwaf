package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafEngine;
import com.aiwaf.spring.annotations.AiwafExempt;
import com.aiwaf.spring.annotations.AiwafExemptFrom;
import com.aiwaf.spring.annotations.AiwafOnly;
import com.aiwaf.spring.annotations.AiwafRequireProtection;
import com.aiwaf.spring.support.AiwafRouteDecisions;
import org.junit.jupiter.api.Test;
import org.springframework.web.method.HandlerMethod;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AiwafRouteDecisionsTest {

    static class DemoController {
        @AiwafExempt
        public void exemptAll() {}

        @AiwafExemptFrom({"rate_limit"})
        public void exemptRate() {}

        @AiwafOnly({"header_validation"})
        public void onlyHeader() {}

        @AiwafRequireProtection({"geo_block"})
        @AiwafExemptFrom({"geo_block"})
        public void requiredWins() {}
    }

    @AiwafExemptFrom({"rate_limit"})
    static class ClassAnnotatedController {
        public void fromClass() {}

        @AiwafOnly({"header_validation"})
        public void classAndMethodOnlyMerge() {}

        @AiwafRequireProtection({"rate_limit"})
        public void requireOverridesClassExempt() {}
    }

    @AiwafExemptFrom({"rate-limit", "uuid-tamper"})
    static class AliasAnnotatedController {
        public void aliases() {}

        @AiwafOnly({"header-validation"})
        public void onlyAlias() {}
    }

    static class AiAnnotatedController {
        @AiwafExemptFrom({"AIAnomalyMiddleware"})
        public void exemptAi() {}
    }

    @Test
    void annotation_based_route_decisions_match_expected() throws Exception {
        DemoController c = new DemoController();
        HandlerMethod exempt = new HandlerMethod(c, DemoController.class.getMethod("exemptAll"));
        HandlerMethod exemptRate = new HandlerMethod(c, DemoController.class.getMethod("exemptRate"));
        HandlerMethod onlyHeader = new HandlerMethod(c, DemoController.class.getMethod("onlyHeader"));
        HandlerMethod requiredWins = new HandlerMethod(c, DemoController.class.getMethod("requiredWins"));

        assertFalse(AiwafRouteDecisions.shouldApply("/x", exempt, "header_validation", List.of()));
        assertFalse(AiwafRouteDecisions.shouldApply("/x", exemptRate, "rate_limit", List.of()));
        assertTrue(AiwafRouteDecisions.shouldApply("/x", exemptRate, "header_validation", List.of()));
        assertTrue(AiwafRouteDecisions.shouldApply("/x", onlyHeader, "header_validation", List.of()));
        assertFalse(AiwafRouteDecisions.shouldApply("/x", onlyHeader, "rate_limit", List.of()));
        assertTrue(AiwafRouteDecisions.shouldApply("/x", requiredWins, "geo_block", List.of()));
    }

    @Test
    void path_rules_disable_when_no_handler_metadata() {
        AiwafConfig.PathRule rule = new AiwafConfig.PathRule(
                "/api/", false, null, null, null, Set.of("rate_limit")
        );
        assertFalse(AiwafRouteDecisions.shouldApply("/api/users", new Object(), "rate_limit", List.of(rule)));
        assertTrue(AiwafRouteDecisions.shouldApply("/api/users", new Object(), "header_validation", List.of(rule)));
    }

    @Test
    void class_and_method_annotations_are_merged_for_selected_middleware_sets() throws Exception {
        ClassAnnotatedController c = new ClassAnnotatedController();
        HandlerMethod fromClass = new HandlerMethod(c, ClassAnnotatedController.class.getMethod("fromClass"));
        HandlerMethod mergedOnly = new HandlerMethod(c, ClassAnnotatedController.class.getMethod("classAndMethodOnlyMerge"));
        HandlerMethod requiredWins = new HandlerMethod(c, ClassAnnotatedController.class.getMethod("requireOverridesClassExempt"));

        assertFalse(AiwafRouteDecisions.shouldApply("/x", fromClass, "rate_limit", List.of()));
        assertTrue(AiwafRouteDecisions.shouldApply("/x", fromClass, "header_validation", List.of()));

        assertFalse(AiwafRouteDecisions.shouldApply("/x", mergedOnly, "rate_limit", List.of()));
        assertTrue(AiwafRouteDecisions.shouldApply("/x", mergedOnly, "header_validation", List.of()));
        assertFalse(AiwafRouteDecisions.shouldApply("/x", mergedOnly, "geo_block", List.of()));

        assertTrue(AiwafRouteDecisions.shouldApply("/x", requiredWins, "rate_limit", List.of()));
    }

    @Test
    void middleware_aliases_in_annotations_are_normalized() throws Exception {
        AliasAnnotatedController c = new AliasAnnotatedController();
        HandlerMethod aliases = new HandlerMethod(c, AliasAnnotatedController.class.getMethod("aliases"));
        HandlerMethod onlyAlias = new HandlerMethod(c, AliasAnnotatedController.class.getMethod("onlyAlias"));

        assertFalse(AiwafRouteDecisions.shouldApply("/x", aliases, "rate_limit", List.of()));
        assertFalse(AiwafRouteDecisions.shouldApply("/x", aliases, "uuid_tamper", List.of()));
        assertTrue(AiwafRouteDecisions.shouldApply("/x", aliases, "header_validation", List.of()));

        assertTrue(AiwafRouteDecisions.shouldApply("/x", onlyAlias, "header_validation", List.of()));
        assertFalse(AiwafRouteDecisions.shouldApply("/x", onlyAlias, "rate_limit", List.of()));
    }

    @Test
    void ai_anomaly_alias_in_annotation_disables_route_scoring() throws Exception {
        AiAnnotatedController controller = new AiAnnotatedController();
        HandlerMethod handler = new HandlerMethod(controller, AiAnnotatedController.class.getMethod("exemptAi"));

        assertFalse(AiwafRouteDecisions.shouldApply("/x", handler, "ai_anomaly", List.of()));
        assertTrue(AiwafRouteDecisions.shouldApply("/x", handler, "rate_limit", List.of()));
    }

    @Test
    void spring_filter_enriches_legitimate_keywords_from_routes() {
        AiwafConfig config = new AiwafConfig();
        AiwafEngine engine = new AiwafEngine(config);
        AiwafFilter filter = new AiwafFilter(engine, new DemoController());
        assertTrue(engine.config().legitimatePathKeywords.contains("exemptall"));
        assertTrue(engine.config().legitimatePathKeywords.contains("onlyheader"));
        assertTrue(filter != null);
    }
}
