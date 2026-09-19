package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RateLimitPolicyCoreParityTest {

    @Test
    void rate_limit_soft_vs_hard_limits() {
        RateLimitPolicyCore.RateLimiter limiter = new RateLimitPolicyCore.RateLimiter(
                new RateLimitPolicyCore.RateLimitConfig(999, 10, 999)
        );
        RateLimitPolicyCore.RequestKey req = new RateLimitPolicyCore.RequestKey("9.9.9.9", "/");
        double t0 = 0.0;

        RateLimitPolicyCore.RateLimitDecision d1 = limiter.registerWithSoftHardLimits(req, t0, 2, 4, 60);
        RateLimitPolicyCore.RateLimitDecision d2 = limiter.registerWithSoftHardLimits(req, t0 + 1, 2, 4, 60);
        RateLimitPolicyCore.RateLimitDecision d3 = limiter.registerWithSoftHardLimits(req, t0 + 2, 2, 4, 60);
        RateLimitPolicyCore.RateLimitDecision d4 = limiter.registerWithSoftHardLimits(req, t0 + 3, 2, 4, 60);
        RateLimitPolicyCore.RateLimitDecision d5 = limiter.registerWithSoftHardLimits(req, t0 + 4, 2, 4, 60);

        assertTrue(d1.allow());
        assertFalse(d1.softLimited());
        assertFalse(d1.hardBlocked());

        assertTrue(d2.allow());
        assertFalse(d2.softLimited());
        assertFalse(d2.hardBlocked());

        assertTrue(d3.allow());
        assertTrue(d3.softLimited());
        assertFalse(d3.hardBlocked());

        assertTrue(d4.allow());
        assertTrue(d4.softLimited());
        assertFalse(d4.hardBlocked());

        assertFalse(d5.allow());
        assertTrue(d5.hardBlocked());
    }

    @Test
    void apply_rate_limit_window_stateless() {
        RateLimitPolicyCore.RateLimitWindowResult result = RateLimitPolicyCore.applyRateLimitWindow(
                List.of(900.0, 995.0),
                1000.0,
                10,
                2,
                4
        );
        assertEquals(List.of(995.0, 1000.0), result.kept());
        assertTrue(result.decision().allow());
        assertFalse(result.decision().softLimited());
        assertFalse(result.decision().hardBlocked());
    }
}
