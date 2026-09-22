package com.aiwaf.runtime;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import redis.clients.jedis.JedisPooled;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.UUID;

/** Redis-backed persistent stores and atomic distributed enforcement state. */
public final class RedisStorage implements StorageBackend, RuntimeState, AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();

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
    private final String prefix;

    public RedisStorage(String redisUrl, String keyPrefix) {
        if (redisUrl == null || redisUrl.isBlank()) {
            throw new IllegalArgumentException("Redis storage requires storageRedisUrl/AIWAF_REDIS_URL");
        }
        this.client = new JedisPooled(URI.create(redisUrl.trim()));
        String configured = keyPrefix == null || keyPrefix.isBlank() ? "aiwaf:" : keyPrefix.trim();
        this.prefix = configured.endsWith(":") ? configured : configured + ":";
    }

    @Override
    public Object get(String key) {
        String raw = client.get(dataKey(key));
        if (raw == null) return null;
        try {
            return JSON.readValue(raw, Object.class);
        } catch (JsonProcessingException ex) {
            throw new IllegalStateException("Invalid JSON stored at Redis key " + key, ex);
        }
    }

    @Override
    public boolean set(String key, Object value, Integer ttlSeconds) {
        try {
            String payload = JSON.writeValueAsString(value);
            if (ttlSeconds == null) {
                client.set(dataKey(key), payload);
            } else {
                client.setex(dataKey(key), Math.max(1, ttlSeconds), payload);
            }
            return true;
        } catch (JsonProcessingException ex) {
            throw new IllegalArgumentException("Value cannot be encoded as JSON", ex);
        }
    }

    @Override
    public boolean delete(String key) {
        return client.del(dataKey(key)) > 0;
    }

    @Override
    public boolean exists(String key) {
        return client.exists(dataKey(key));
    }

    @Override
    public List<String> getAllKeys(String globPattern) {
        String pattern = globPattern == null || globPattern.isBlank() ? "*" : globPattern;
        Set<String> keys = client.keys(dataKey(pattern));
        List<String> out = new ArrayList<>();
        String dataPrefix = prefix + "data:";
        for (String key : keys) {
            if (key.startsWith(dataPrefix)) out.add(key.substring(dataPrefix.length()));
        }
        return out;
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
    }

    @Override
    public boolean recordFormGet(String key, long nowMillis, int ttlSeconds, int maxEntries) {
        // Retain expired form views long enough to report page_expired instead of
        // silently treating the later POST as a form that was never observed.
        client.setex(stateKey("form", key), Math.max(3_600, ttlSeconds * 2), String.valueOf(nowMillis));
        return true;
    }

    @Override
    public Long getFormGet(String key) {
        String value = client.get(stateKey("form", key));
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
        client.eval(
                RECENT_RECORD_SCRIPT,
                1,
                stateKey("recent", subject),
                String.valueOf(nowMillis),
                String.valueOf(nowMillis - windowMillis),
                String.valueOf(windowMillis),
                member
        );
    }

    @Override
    public RecentStats recentStats(String subject, long nowMillis, int windowSeconds) {
        long windowMillis = Math.max(1, windowSeconds) * 1_000L;
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
    }

    @Override
    public void clear() {
        for (String key : client.keys(prefix + "state:*")) client.del(key);
    }

    @Override
    public void close() {
        client.close();
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
