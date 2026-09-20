package com.aiwaf.runtime;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class BlacklistRuntimeParityTest {

    @BeforeEach
    void setup() {
        RuntimeStorage.initialize("memory", null);
    }

    @Test
    void block_and_unblock_flow() {
        assertTrue(BlacklistManager.block("8.8.8.8", "test", 60));
        assertTrue(BlacklistManager.isBlocked("8.8.8.8"));
        assertNotNull(BlacklistManager.getBlockInfo("8.8.8.8"));
        assertTrue(BlacklistManager.unblock("8.8.8.8"));
        assertFalse(BlacklistManager.isBlocked("8.8.8.8"));
    }

    @Test
    void default_blocks_use_python_reputation_and_progressive_durations() {
        assertTrue(BlacklistManager.block("8.8.4.4", "SQL injection"));
        Map<String, Object> first = BlacklistManager.getBlockInfo("8.8.4.4");
        assertEquals(40, ((Number) first.get("score")).intValue());
        assertEquals(1, ((Number) first.get("offenses")).intValue());
        assertEquals(900, ((Number) first.get("duration")).intValue());
        assertFalse((Boolean) first.get("permanent"));

        assertTrue(BlacklistManager.block("8.8.4.4", "XSS"));
        Map<String, Object> second = BlacklistManager.getBlockInfo("8.8.4.4");
        assertEquals(70, ((Number) second.get("score")).intValue());
        assertEquals(2, ((Number) second.get("offenses")).intValue());
        assertEquals(3600, ((Number) second.get("duration")).intValue());

        assertTrue(BlacklistManager.block("8.8.4.4", "Header validation"));
        Map<String, Object> third = BlacklistManager.getBlockInfo("8.8.4.4");
        assertEquals(85, ((Number) third.get("score")).intValue());
        assertEquals(3, ((Number) third.get("offenses")).intValue());
        assertEquals(86400, ((Number) third.get("duration")).intValue());
    }

    @Test
    void explicit_permanent_block_remains_available() {
        assertTrue(BlacklistManager.blockPermanent("4.4.4.4", "manual"));
        Map<String, Object> block = BlacklistManager.getBlockInfo("4.4.4.4");
        assertTrue((Boolean) block.get("permanent"));
        assertNull(block.get("duration"));
    }

    @Test
    void whitelist_prevents_blocking() {
        BlacklistManager.addToWhitelist("1.2.3.4", "manual");
        assertFalse(BlacklistManager.block("1.2.3.4", "should not block", 60));
        assertFalse(BlacklistManager.isBlocked("1.2.3.4"));
        assertTrue(BlacklistManager.isWhitelisted("1.2.3.4"));
    }

    @Test
    void stats_and_recent_blocks_are_recorded() {
        BlacklistManager.block("9.9.9.9", "reason-a", null);
        BlacklistManager.block("7.7.7.7", "reason-b", null);
        Map<String, Object> stats = BlacklistManager.getStatistics();
        assertTrue(((Number) stats.get("total_blocked")).intValue() >= 2);
        assertFalse(BlacklistManager.getRecentBlocks(24).isEmpty());
        assertFalse(BlacklistManager.getTopBlockedReasons(10).isEmpty());
    }
}
