package com.aiwaf.runtime;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;
import redis.clients.jedis.Connection;
import redis.clients.jedis.JedisPooled;
import redis.clients.jedis.params.ScanParams;
import redis.clients.jedis.resps.ScanResult;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;

/** Redis-backed persistent stores and atomic distributed enforcement state. */
public final class RedisStorage implements StorageBackend, RuntimeState, AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();
    public static final String STATE_SCHEMA_VERSION = "1";

    public enum FailureMode {
        FAIL_CLOSED,
        FAIL_OPEN;

        public static FailureMode parse(String value) {
            if (value == null || value.isBlank()) return FAIL_CLOSED;
            return switch (value.trim().toLowerCase()) {
                case "fail_closed", "fail-closed", "closed" -> FAIL_CLOSED;
                case "fail_open", "fail-open", "open" -> FAIL_OPEN;
                default -> throw new IllegalArgumentException(
                        "Redis failure mode must be fail_closed or fail_open");
            };
        }
    }

    /** Bounded standalone Redis client settings. Schema changes must be explicit. */
    public record Options(
            String mode,
            FailureMode failureMode,
            int connectionTimeoutMillis,
            int socketTimeoutMillis,
            int poolMaxTotal,
            int poolMaxIdle,
            int poolMinIdle,
            long poolMaxWaitMillis
    ) {
        public Options {
            mode = mode == null || mode.isBlank() ? "standalone" : mode.trim().toLowerCase();
            failureMode = failureMode == null ? FailureMode.FAIL_CLOSED : failureMode;
            if (!"standalone".equals(mode)) {
                throw new IllegalArgumentException(
                        "Redis mode '" + mode + "' is unsupported; Java 1.3 supports standalone Redis only");
            }
            if (connectionTimeoutMillis < 1 || socketTimeoutMillis < 1 || poolMaxWaitMillis < 1) {
                throw new IllegalArgumentException("Redis timeouts and pool max wait must be positive");
            }
            if (poolMaxTotal < 1 || poolMaxIdle < 0 || poolMinIdle < 0
                    || poolMaxIdle > poolMaxTotal || poolMinIdle > poolMaxIdle) {
                throw new IllegalArgumentException("Invalid Redis pool bounds");
            }
        }

        public static Options defaults() {
            return new Options("standalone", FailureMode.FAIL_CLOSED, 1_000, 2_000,
                    32, 16, 0, 250);
        }
    }

    /** Lightweight operation, latency, failure, and pool telemetry. */
    public record Diagnostics(
            long operationCount,
            long failureCount,
            long totalLatencyNanos,
            long maxLatencyNanos,
            int poolActive,
            int poolIdle,
            int poolWaiters
    ) {
        public double averageLatencyMillis() {
            return operationCount == 0 ? 0.0 : totalLatencyNanos / 1_000_000.0 / operationCount;
        }
    }

    public static final class RedisUnavailableException extends IllegalStateException {
        RedisUnavailableException(String message, Throwable cause) { super(message, cause); }
    }

    public static final class RedisStateCompatibilityException extends IllegalStateException {
        RedisStateCompatibilityException(String message) { super(message); }
    }

    private static final String RATE_SCRIPT = """
            redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[2])
            redis.call('ZADD', KEYS[1], ARGV[1], ARGV[6])
            redis.call('PEXPIRE', KEYS[1], ARGV[3])
            local count = redis.call('ZCARD', KEYS[1])
            if count > tonumber(ARGV[5]) then return {2, count} end
            if count > tonumber(ARGV[4]) then return {1, count} end
            return {0, count}
            """;

    private static final String UUID_SCRIPT = """
            redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[2])
            redis.call('ZADD', KEYS[1], ARGV[1], ARGV[6])
            redis.call('PEXPIRE', KEYS[1], ARGV[3])
            local entries = redis.call('ZRANGE', KEYS[1], 0, -1)
            local score = 0
            for _, entry in ipairs(entries) do
              local first = string.find(entry, '|', 1, true)
              local second = first and string.find(entry, '|', first + 1, true)
              if first and second then score = score + tonumber(string.sub(entry, first + 1, second - 1)) end
            end
            local blocked = 0
            if score >= tonumber(ARGV[5]) then blocked = 1 end
            return {score, blocked}
            """;

    private static final String RECENT_RECORD_SCRIPT = """
            redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[2])
            redis.call('ZADD', KEYS[1], ARGV[1], ARGV[4])
            redis.call('PEXPIRE', KEYS[1], ARGV[3])
            return 1
            """;

    private static final String RECENT_STATS_SCRIPT = """
            redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[2])
            local entries = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
            local count = 0
            local notFound = 0
            local burst = 0
            for i = 1, #entries, 2 do
              count = count + 1
              local entry = entries[i]
              local first = string.find(entry, '|', 1, true)
              local second = first and string.find(entry, '|', first + 1, true)
              if first and second and tonumber(string.sub(entry, first + 1, second - 1)) == 404 then
                notFound = notFound + 1
              end
              if tonumber(entries[i + 1]) >= tonumber(ARGV[3]) then burst = burst + 1 end
            end
            return {count, notFound, burst}
            """;

    private final JedisPooled client;
    private final Options options;
    private final String stateSchemaVersion;
    private final String basePrefix;
    private final String prefix;
    private final AtomicBoolean schemaVerified = new AtomicBoolean();
    private final LongAdder operationCount = new LongAdder();
    private final LongAdder failureCount = new LongAdder();
    private final LongAdder totalLatencyNanos = new LongAdder();
    private final AtomicLong maxLatencyNanos = new AtomicLong();

    public RedisStorage(String redisUrl, String keyPrefix) {
        this(redisUrl, keyPrefix, Options.defaults());
    }

    public RedisStorage(String redisUrl, String keyPrefix, Options options) {
        this(redisUrl, keyPrefix, options, STATE_SCHEMA_VERSION);
    }

    RedisStorage(String redisUrl, String keyPrefix, Options options, String stateSchemaVersion) {
        if (redisUrl == null || redisUrl.isBlank()) {
            throw new IllegalArgumentException("Redis storage requires storageRedisUrl/AIWAF_REDIS_URL");
        }
        URI uri = URI.create(redisUrl.trim());
        if (!"redis".equalsIgnoreCase(uri.getScheme()) && !"rediss".equalsIgnoreCase(uri.getScheme())) {
            throw new IllegalArgumentException("Redis URL must use redis:// or rediss://");
        }
        this.options = options == null ? Options.defaults() : options;
        this.stateSchemaVersion = stateSchemaVersion == null ? "" : stateSchemaVersion.trim();
        if (!this.stateSchemaVersion.matches("[A-Za-z0-9._-]{1,32}")) {
            throw new IllegalArgumentException("Invalid Redis state schema version");
        }
        GenericObjectPoolConfig<Connection> pool = new GenericObjectPoolConfig<>();
        pool.setMaxTotal(this.options.poolMaxTotal());
        pool.setMaxIdle(this.options.poolMaxIdle());
        pool.setMinIdle(this.options.poolMinIdle());
        pool.setBlockWhenExhausted(true);
        pool.setMaxWaitMillis(this.options.poolMaxWaitMillis());
        pool.setJmxEnabled(false);
        this.client = new JedisPooled(
                pool,
                uri,
                this.options.connectionTimeoutMillis(),
                this.options.socketTimeoutMillis()
        );
        String configured = keyPrefix == null || keyPrefix.isBlank() ? "aiwaf:" : keyPrefix.trim();
        this.basePrefix = configured.endsWith(":") ? configured : configured + ":";
        this.prefix = basePrefix + "v" + this.stateSchemaVersion + ":";
        try {
            execute("startup", () -> null, () -> null);
        } catch (RuntimeException ex) {
            client.close();
            throw ex;
        }
    }

    @Override
    public Object get(String key) {
        String raw = execute("get", () -> client.get(dataKey(key)), () -> null);
        if (raw == null) return null;
        try {
            return JSON.readValue(raw, Object.class);
        } catch (JsonProcessingException ex) {
            throw new IllegalStateException("Invalid JSON stored at Redis key " + key, ex);
        }
    }

    @Override
    public boolean set(String key, Object value, Integer ttlSeconds) {
        final String payload;
        try {
            payload = JSON.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new IllegalArgumentException("Value cannot be encoded as JSON", ex);
        }
        return execute("set", () -> {
            if (ttlSeconds == null) {
                client.set(dataKey(key), payload);
            } else {
                client.setex(dataKey(key), Math.max(1, ttlSeconds), payload);
            }
            return true;
        }, () -> false);
    }

    @Override
    public boolean delete(String key) {
        return execute("delete", () -> client.del(dataKey(key)) > 0, () -> false);
    }

    @Override
    public boolean exists(String key) {
        return execute("exists", () -> client.exists(dataKey(key)), () -> false);
    }

    @Override
    public List<String> getAllKeys(String globPattern) {
        String pattern = globPattern == null || globPattern.isBlank() ? "*" : globPattern;
        return execute("scan", () -> {
            List<String> out = new ArrayList<>();
            String dataPrefix = prefix + "data:";
            for (String key : scanKeys(dataKey(pattern))) {
                if (key.startsWith(dataPrefix)) out.add(key.substring(dataPrefix.length()));
            }
            Collections.sort(out);
            return out;
        }, List::of);
    }

    @Override
    public RateResult recordRate(
            String key,
            long nowMillis,
            int windowSeconds,
            int maxRequests,
            int floodThreshold,
            int maxEntries
    ) {
        long windowMillis = Math.max(1, windowSeconds) * 1_000L;
        return execute("record_rate", () -> {
            Object result = client.eval(
                    RATE_SCRIPT,
                    1,
                    stateKey("rate", key),
                    String.valueOf(nowMillis),
                    String.valueOf(nowMillis - windowMillis),
                    String.valueOf(windowMillis),
                    String.valueOf(Math.max(1, maxRequests)),
                    String.valueOf(Math.max(1, floodThreshold)),
                    nowMillis + "|" + UUID.randomUUID()
            );
            List<?> values = asList(result);
            int action = number(values, 0);
            RateAction rateAction = action == 2 ? RateAction.FLOOD : action == 1 ? RateAction.LIMIT : RateAction.ALLOW;
            return new RateResult(rateAction, number(values, 1));
        }, () -> new RateResult(RateAction.ALLOW, 0));
    }

    @Override
    public boolean recordFormGet(String key, long nowMillis, int ttlSeconds, int maxEntries) {
        // Retain expired form views long enough to report page_expired instead of
        // silently treating the later POST as a form that was never observed.
        return execute("record_form_get", () -> {
            client.setex(stateKey("form", key), Math.max(3_600, ttlSeconds * 2), String.valueOf(nowMillis));
            return true;
        }, () -> false);
    }

    @Override
    public Long getFormGet(String key) {
        String value = execute("get_form_get", () -> client.get(stateKey("form", key)), () -> null);
        if (value == null) return null;
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    @Override
    public UuidResult recordUuid(
            String subject,
            int delta,
            long nowMillis,
            int windowSeconds,
            int blockThreshold,
            int maxEntries
    ) {
        if (delta == 0) return new UuidResult(0, false);
        long windowMillis = Math.max(1, windowSeconds) * 1_000L;
        String member = nowMillis + "|" + delta + "|" + UUID.randomUUID();
        return execute("record_uuid", () -> {
            Object result = client.eval(
                    UUID_SCRIPT,
                    1,
                    stateKey("uuid", subject),
                    String.valueOf(nowMillis),
                    String.valueOf(nowMillis - windowMillis),
                    String.valueOf(windowMillis),
                    String.valueOf(delta),
                    String.valueOf(Math.max(1, blockThreshold)),
                    member
            );
            List<?> values = asList(result);
            return new UuidResult(number(values, 0), number(values, 1) == 1);
        }, () -> new UuidResult(0, false));
    }

    @Override
    public void recordRecent(
            String subject,
            long nowMillis,
            int statusCode,
            int windowSeconds,
            int maxEntries
    ) {
        long windowMillis = Math.max(1, windowSeconds) * 1_000L;
        String member = nowMillis + "|" + statusCode + "|" + UUID.randomUUID();
        execute("record_recent", () -> {
            client.eval(
                    RECENT_RECORD_SCRIPT,
                    1,
                    stateKey("recent", subject),
                    String.valueOf(nowMillis),
                    String.valueOf(nowMillis - windowMillis),
                    String.valueOf(windowMillis),
                    member
            );
            return null;
        }, () -> null);
    }

    @Override
    public RecentStats recentStats(String subject, long nowMillis, int windowSeconds) {
        long windowMillis = Math.max(1, windowSeconds) * 1_000L;
        return execute("recent_stats", () -> {
            Object result = client.eval(
                    RECENT_STATS_SCRIPT,
                    1,
                    stateKey("recent", subject),
                    String.valueOf(nowMillis),
                    String.valueOf(nowMillis - windowMillis),
                    String.valueOf(nowMillis - 10_000L)
            );
            List<?> values = asList(result);
            return new RecentStats(number(values, 0), number(values, 1), number(values, 2));
        }, () -> new RecentStats(0, 0, 0));
    }

    @Override
    public void clear() {
        execute("clear", () -> {
            List<String> keys = scanKeys(prefix + "state:*");
            if (!keys.isEmpty()) client.del(keys.toArray(String[]::new));
            return null;
        }, () -> null);
    }

    @Override
    public void close() {
        client.close();
    }

    public Diagnostics diagnostics() {
        return new Diagnostics(
                operationCount.sum(),
                failureCount.sum(),
                totalLatencyNanos.sum(),
                maxLatencyNanos.get(),
                client.getPool().getNumActive(),
                client.getPool().getNumIdle(),
                client.getPool().getNumWaiters()
        );
    }

    public String stateSchemaVersion() {
        return stateSchemaVersion;
    }

    public FailureMode failureMode() {
        return options.failureMode();
    }

    AutoCloseable holdConnectionForTest() {
        Connection connection = client.getPool().getResource();
        return () -> client.getPool().returnResource(connection);
    }

    private void ensureCompatibleSchema() {
        if (schemaVerified.get()) return;
        synchronized (schemaVerified) {
            if (schemaVerified.get()) return;
            String markerKey = basePrefix + "meta:state-schema";
            long created = client.setnx(markerKey, stateSchemaVersion);
            String actual = created == 1L ? stateSchemaVersion : client.get(markerKey);
            if (!stateSchemaVersion.equals(actual)) {
                throw new RedisStateCompatibilityException(
                        "Redis namespace '" + basePrefix + "' uses state schema '" + actual
                                + "', but this AIWAF runtime requires '" + stateSchemaVersion
                                + "'. Use a different storage key prefix for incompatible versions.");
            }
            schemaVerified.set(true);
        }
    }

    private List<String> scanKeys(String pattern) {
        List<String> keys = new ArrayList<>();
        ScanParams params = new ScanParams().match(pattern).count(250);
        String cursor = ScanParams.SCAN_POINTER_START;
        do {
            ScanResult<String> page = client.scan(cursor, params);
            keys.addAll(page.getResult());
            cursor = page.getCursor();
        } while (!ScanParams.SCAN_POINTER_START.equals(cursor));
        return keys;
    }

    private <T> T execute(String operation, Supplier<T> action, Supplier<T> failOpenFallback) {
        long started = System.nanoTime();
        operationCount.increment();
        try {
            ensureCompatibleSchema();
            return action.get();
        } catch (RedisStateCompatibilityException ex) {
            failureCount.increment();
            throw ex;
        } catch (RuntimeException ex) {
            failureCount.increment();
            if (options.failureMode() == FailureMode.FAIL_OPEN) return failOpenFallback.get();
            throw new RedisUnavailableException("Redis operation '" + operation + "' failed", ex);
        } finally {
            long elapsed = System.nanoTime() - started;
            totalLatencyNanos.add(elapsed);
            maxLatencyNanos.accumulateAndGet(elapsed, Math::max);
        }
    }

    private String dataKey(String key) {
        return prefix + "data:" + key;
    }

    private String stateKey(String kind, String logicalKey) {
        return prefix + "state:" + kind + ":" + sha256(logicalKey == null ? "" : logicalKey);
    }

    private static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 unavailable", ex);
        }
    }

    private static List<?> asList(Object value) {
        return value instanceof List<?> values ? values : List.of();
    }

    private static int number(List<?> values, int index) {
        if (index >= values.size()) return 0;
        Object value = values.get(index);
        if (value instanceof Number number) return number.intValue();
        try {
            return Integer.parseInt(String.valueOf(value));
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }
}
