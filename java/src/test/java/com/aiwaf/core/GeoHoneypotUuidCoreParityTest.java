package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

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

    @Test
    void uuid_score_matches_python_weights_decay_and_window() {
        UuidScoreCore scores = new UuidScoreCore();
        UuidScoreCore.Config config = new UuidScoreCore.Config(true, 60, 5, 5, 1, 2);

        assertEquals(1, scores.record("ip", UuidScoreCore.Signal.NOT_FOUND, 1_000, config).score());
        assertEquals(2, scores.record("ip", UuidScoreCore.Signal.NOT_FOUND, 2_000, config).score());
        assertEquals(0, scores.record("ip", UuidScoreCore.Signal.SUCCESS, 3_000, config).score());
        assertTrue(scores.record("ip", UuidScoreCore.Signal.MALFORMED, 4_000, config).blocked());
        assertEquals(1, scores.record("ip", UuidScoreCore.Signal.NOT_FOUND, 70_000, config).score());
    }

    @Test
    void uuid_model_field_lookup_skips_bad_field_and_finds_later_match() {
        List<UuidPolicyCore.UUIDModelField> fields = List.of(
                new UuidPolicyCore.UUIDModelField(String.class, "first", true),
                new UuidPolicyCore.UUIDModelField(Integer.class, "second", false)
        );
        UUID expected = UUID.fromString("123e4567-e89b-12d3-a456-426614174000");
        assertTrue(UuidPolicyCore.uuidExistsInModelFields(expected.toString(), fields, (model, field, value) -> {
            if (field.equals("first")) throw new IllegalArgumentException("wrong model");
            return field.equals("second") && value.equals(expected);
        }));
    }
}
