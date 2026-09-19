package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafDecision;
import com.aiwaf.core.AiwafEngine;
import com.aiwaf.core.AiwafRequest;
import com.aiwaf.core.FastRModelImportCore;
import com.aiwaf.core.ModelArtifactIoCore;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AiwafPostResponseTest {
    @TempDir Path tempDir;

    @Test
    void actual_elapsed_time_changes_post_response_ai_decision() throws Exception {
        AiwafEngine engine = engineForFeature(2, 0.02);
        AiwafRequest fast = request("198.51.100.201");
        AiwafRequest slow = request("198.51.100.202");

        assertTrue(engine.evaluateBeforeResponse(fast).allowed());
        assertTrue(engine.evaluateAfterResponse(fast, 200, 5).allowed());
        assertTrue(engine.evaluateBeforeResponse(slow).allowed());
        assertEquals(403, engine.evaluateAfterResponse(slow, 200, 50).statusCode());
    }

    @Test
    void actual_handler_status_changes_post_response_ai_decision() throws Exception {
        AiwafEngine engine = engineForFeature(3, 5.0);
        AiwafRequest ok = request("198.51.100.203");
        AiwafRequest missing = request("198.51.100.204");

        assertTrue(engine.evaluateBeforeResponse(ok).allowed());
        assertTrue(engine.evaluateAfterResponse(ok, 200, 5).allowed());
        assertTrue(engine.evaluateBeforeResponse(missing).allowed());
        assertEquals(403, engine.evaluateAfterResponse(missing, 404, 5).statusCode());
    }

    @Test
    void spring_filter_replaces_uncommitted_anomalous_response() throws Exception {
        AiwafFilter filter = new AiwafFilter(engineForFeature(3, 5.0));
        MockHttpServletRequest request = servletRequest("198.51.100.205");
        MockHttpServletResponse response = new MockHttpServletResponse();

        filter.doFilter(request, response, (req, res) -> {
            res.setContentType("text/plain");
            ((MockHttpServletResponse) res).setStatus(404);
            res.getWriter().write("original response");
        });

        assertEquals(403, response.getStatus());
    }

    @Test
    void spring_filter_preserves_committed_streaming_response() throws Exception {
        AiwafFilter filter = new AiwafFilter(engineForFeature(3, 5.0));
        MockHttpServletRequest request = servletRequest("198.51.100.206");
        MockHttpServletResponse response = new MockHttpServletResponse();

        filter.doFilter(request, response, (req, res) -> {
            ((MockHttpServletResponse) res).setStatus(404);
            res.getWriter().write("stream chunk");
            res.flushBuffer();
        });

        assertEquals(404, response.getStatus());
        assertEquals("stream chunk", response.getContentAsString());
    }

    @Test
    void committed_response_still_updates_actual_status_history() throws Exception {
        AiwafFilter filter = new AiwafFilter(engineForFeature(5, 0.5));
        String ip = "198.51.100.207";
        MockHttpServletResponse committed = new MockHttpServletResponse();
        filter.doFilter(servletRequest(ip), committed, (req, res) -> {
            ((MockHttpServletResponse) res).setStatus(404);
            res.flushBuffer();
        });
        assertEquals(404, committed.getStatus());

        MockHttpServletResponse next = new MockHttpServletResponse();
        filter.doFilter(servletRequest(ip), next, (req, res) -> {
            ((MockHttpServletResponse) res).setStatus(200);
        });
        assertEquals(403, next.getStatus());
    }

    @Test
    void asynchronous_response_is_not_replaced_after_handler_returns() throws Exception {
        AiwafFilter filter = new AiwafFilter(engineForFeature(3, 5.0));
        MockHttpServletRequest request = servletRequest("198.51.100.208");
        request.setAsyncSupported(true);
        MockHttpServletResponse response = new MockHttpServletResponse();

        filter.doFilter(request, response, (req, res) -> {
            ((MockHttpServletResponse) res).setStatus(404);
            req.startAsync(req, res);
        });

        assertTrue(request.isAsyncStarted());
        assertEquals(404, response.getStatus());
        request.getAsyncContext().complete();
        assertFalse(request.isAsyncStarted());
        assertEquals(404, response.getStatus());
    }

    private AiwafEngine engineForFeature(int featureIndex, double split) throws Exception {
        String json = """
                {
                  "model_type":"isolation-forest",
                  "model_schema":"iforest-v1",
                  "backend":"fastr_aiwaf",
                  "feature_names":["path_len","kw_hits","resp_time","status_idx","burst_count","total_404"],
                  "static_keywords":[],
                  "sample_size":6,
                  "trees":[{"feature_subset":[%d],"root":{
                    "leaf":false,"feature":%d,"split":%s,
                    "left":{"leaf":true,"leaf_size":5},
                    "right":{"leaf":true,"leaf_size":1}
                  }}],
                  "threshold":0.6,
                  "metadata":{"feature_schema":"python-six-v1"}
                }
                """.formatted(featureIndex, featureIndex, split);
        Path modelPath = tempDir.resolve("model-" + featureIndex + ".bin");
        assertTrue(ModelArtifactIoCore.save(FastRModelImportCore.parse(json), modelPath.toString()));

        AiwafConfig config = new AiwafConfig();
        config.aiEnabled = true;
        config.aiModelPath = modelPath.toString();
        config.aiRequireBehaviorConfirmation = false;
        config.privateIpsExempted = false;
        config.headerValidationEnabled = false;
        config.rateLimitEnabled = false;
        config.ipKeywordBlockEnabled = false;
        return new AiwafEngine(config);
    }

    private static AiwafRequest request(String ip) {
        return new AiwafRequest("GET", "/profile", ip, "", Map.of(), Map.of(),
                System.currentTimeMillis(), Set.of());
    }

    private static MockHttpServletRequest servletRequest(String ip) {
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/profile");
        request.setRemoteAddr(ip);
        return request;
    }
}
