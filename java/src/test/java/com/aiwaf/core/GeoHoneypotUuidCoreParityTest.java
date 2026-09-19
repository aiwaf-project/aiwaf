package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GeoHoneypotUuidCoreParityTest {

    @Test
    void geo_block_allowlist_mode() {
        assertFalse(GeoCore.shouldGeoBlock("US", List.of("US", "CA"), List.of()));
        assertTrue(GeoCore.shouldGeoBlock("RU", List.of("US", "CA"), List.of()));
    }

    @Test
    void geo_block_blocklist_mode() {
        assertTrue(GeoCore.shouldGeoBlock("RU", List.of(), List.of("RU", "CN")));
        assertFalse(GeoCore.shouldGeoBlock("US", List.of(), List.of("RU", "CN")));
    }

    @Test
    void honeypot_fast_post_blocked() {
        HoneypotCore.HoneypotDecision decision = HoneypotCore.evaluateHoneypotRequest(
                "POST",
                "/contact/submit/",
                10.0,
                9.5,
                new HoneypotCore.HoneypotConfig(),
                false,
                true
        );
        assertFalse(decision.allow());
        assertEquals(403, decision.statusCode());
    }

    @Test
    void honeypot_page_expired() {
        HoneypotCore.HoneypotConfig cfg = new HoneypotCore.HoneypotConfig(
                1.0,
                5.0,
                0.1,
                new HoneypotCore.HoneypotConfig().loginPrefixes()
        );
        HoneypotCore.HoneypotDecision decision = HoneypotCore.evaluateHoneypotRequest(
                "POST",
                "/contact/submit/",
                20.0,
                10.0,
                cfg,
                false,
                true
        );
        assertFalse(decision.allow());
        assertEquals(409, decision.statusCode());
        assertTrue(decision.reloadRequired());
    }

    @Test
    void block_policy_respects_exemption() {
        assertFalse(BlockPolicyCore.canBlockIp(true, new BlockPolicyCore.BlockPolicyConfig(true)));
        assertTrue(BlockPolicyCore.canBlockIp(false, new BlockPolicyCore.BlockPolicyConfig(true)));
    }

    @Test
    void uuid_policy_invalid_and_not_found() {
        UuidPolicyCore.UUIDTamperDecision bad = UuidPolicyCore.evaluateUuidTamper("not-a-uuid", null);
        assertFalse(bad.allow());
        assertEquals("invalid_uuid_format", bad.reason());

        UuidPolicyCore.UUIDTamperDecision missing = UuidPolicyCore.evaluateUuidTamper(
                "123e4567-e89b-12d3-a456-426614174000",
                value -> false
        );
        assertFalse(missing.allow());
        assertEquals("uuid_not_found", missing.reason());
    }
}
