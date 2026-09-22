package com.aiwaf.runtime;

import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class RedisStorageIntegrationTest {
    @Test
    void shares_atomic_enforcement_and_persistent_stores_across_contexts() {
        String url = System.getenv("AIWAF_TEST_REDIS_URL");
        assumeTrue(url != null && !url.isBlank(), "AIWAF_TEST_REDIS_URL is not configured");
        String prefix = "aiwaf-test-" + UUID.randomUUID();

        RuntimeStorage.Context first = RuntimeStorage.create("redis", null, url, prefix);
        RuntimeStorage.Context second = RuntimeStorage.create("redis", null, url, prefix);
        try {
            assertEquals(RuntimeState.RateAction.ALLOW,
                    first.state().recordRate("client|route", 1_000, 60, 1, 10, 100).action());
            assertEquals(RuntimeState.RateAction.LIMIT,
                    second.state().recordRate("client|route", 1_001, 60, 1, 10, 100).action());

            first.exemptionStore().addIp("198.51.100.80", "integration test");
            assertTrue(second.exemptionStore().isExempted("198.51.100.80"));
        } finally {
            first.state().clear();
            first.storage().delete("exemptions");
            ((RedisStorage) first.storage()).close();
            ((RedisStorage) second.storage()).close();
        }
    }
}
