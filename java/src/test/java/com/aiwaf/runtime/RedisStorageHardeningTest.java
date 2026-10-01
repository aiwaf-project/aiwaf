package com.aiwaf.runtime;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RedisStorageHardeningTest {
    @Test
    void fail_open_has_bounded_safe_fallbacks_when_redis_is_unavailable() {
        RedisStorage storage = new RedisStorage(
                "redis://127.0.0.1:1",
                "unavailable-test",
                options(RedisStorage.FailureMode.FAIL_OPEN, 50, 1, 50)
        );
        try {
            long started = System.nanoTime();
            RuntimeState.RateResult rate = storage.recordRate("client", 1_000, 60, 1, 2, 100);
            long elapsedMillis = (System.nanoTime() - started) / 1_000_000L;

            assertEquals(RuntimeState.RateAction.ALLOW, rate.action());
            assertFalse(storage.exists("missing"));
            assertTrue(elapsedMillis < 2_000, "Redis fallback must be bounded by configured timeouts");
            assertTrue(storage.diagnostics().failureCount() >= 2);
        } finally {
            storage.close();
        }
    }

    @Test
    void fail_closed_rejects_startup_when_redis_is_unavailable() {
        assertThrows(RedisStorage.RedisUnavailableException.class, () -> new RedisStorage(
                "redis://127.0.0.1:1",
                "unavailable-test",
                options(RedisStorage.FailureMode.FAIL_CLOSED, 50, 1, 50)
        ));
    }

    @Test
    void rejects_unsupported_topologies_and_unbounded_pool_settings() {
        assertThrows(IllegalArgumentException.class, () -> new RedisStorage.Options(
                "cluster", RedisStorage.FailureMode.FAIL_CLOSED, 100, 100,
                10, 5, 0, 100));
        assertThrows(IllegalArgumentException.class, () -> new RedisStorage.Options(
                "standalone", RedisStorage.FailureMode.FAIL_CLOSED, 100, 100,
                1, 2, 0, 100));
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
