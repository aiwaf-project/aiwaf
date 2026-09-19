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
