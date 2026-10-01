package com.aiwaf.runtime;

import org.junit.jupiter.api.Test;

import java.util.UUID;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
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

            first.storage().set("rolling-upgrade", Map.of("schema", 1), null);
            assertEquals(Map.of("schema", 1), second.storage().get("rolling-upgrade"));
            RedisStorage firstRedis = (RedisStorage) first.storage();
            assertEquals(RedisStorage.STATE_SCHEMA_VERSION, firstRedis.stateSchemaVersion());
            assertTrue(firstRedis.diagnostics().operationCount() > 0);

            RedisStorage.Options incompatible = options(
                    RedisStorage.FailureMode.FAIL_CLOSED, 1_000, 2, 100);
            assertThrows(RedisStorage.RedisStateCompatibilityException.class,
                    () -> new RedisStorage(url, prefix, incompatible, "2"));
        } finally {
            first.state().clear();
            first.storage().delete("exemptions");
            first.storage().delete("rolling-upgrade");
            ((RedisStorage) first.storage()).close();
            ((RedisStorage) second.storage()).close();
        }
    }

    @Test
    void pool_exhaustion_is_bounded_and_observable_in_fail_open_mode() throws Exception {
        String url = System.getenv("AIWAF_TEST_REDIS_URL");
        assumeTrue(url != null && !url.isBlank(), "AIWAF_TEST_REDIS_URL is not configured");
        RedisStorage storage = new RedisStorage(
                url,
                "aiwaf-pool-test-" + UUID.randomUUID(),
                options(RedisStorage.FailureMode.FAIL_OPEN, 1_000, 1, 75)
        );
        try (AutoCloseable ignored = storage.holdConnectionForTest()) {
            long started = System.nanoTime();
            assertFalse(storage.exists("missing"));
            long elapsedMillis = (System.nanoTime() - started) / 1_000_000L;
            assertTrue(elapsedMillis >= 50 && elapsedMillis < 1_000,
                    "pool exhaustion should honor the bounded max wait");
            assertTrue(storage.diagnostics().failureCount() >= 1);
        } finally {
            storage.close();
        }
    }

    private static RedisStorage.Options options(
            RedisStorage.FailureMode failureMode,
            int timeoutMillis,
            int poolMaxTotal,
            long poolMaxWaitMillis
    ) {
        return new RedisStorage.Options(
                "standalone", failureMode, timeoutMillis, timeoutMillis,
                poolMaxTotal, poolMaxTotal, 0, poolMaxWaitMillis);
    }
}
