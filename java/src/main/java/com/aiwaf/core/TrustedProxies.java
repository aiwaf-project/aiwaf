package com.aiwaf.core;

import com.aiwaf.runtime.CidrUtil;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/** Trusted proxy addresses: shortcut names, address checks and the misconfiguration warning. */
public final class TrustedProxies {
    private static final Logger LOGGER = Logger.getLogger(TrustedProxies.class.getName());

    /** Java's built-in trusted proxies, used when none are configured: loopback only. */
    public static final List<String> DEFAULT = List.of("127.0.0.0/8", "::1/128");

    /**
     * Published at https://www.cloudflare.com/ips/ (retrieved 2026-11-15). Keep in step with
     * CLOUDFLARE_PROXIES in py/aiwaf/core/utils.py and js/lib/runtimeUtils.js. Opt-in only:
     * anyone can send requests through Cloudflare, so these ranges don't prove a request
     * came from your own zone.
     */
    public static final List<String> CLOUDFLARE = List.of(
            "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22",
            "141.101.64.0/18", "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20",
            "197.234.240.0/22", "198.41.128.0/17", "162.158.0.0/15", "104.16.0.0/13",
            "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
            "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32", "2405:b500::/32",
            "2405:8100::/32", "2a06:98c0::/29", "2c0f:f248::/32");

    /** Names accepted alongside addresses and ranges, for example "default,cloudflare". */
    private static final Map<String, List<String>> PRESETS = Map.of("default", DEFAULT, "cloudflare", CLOUDFLARE);

    private static final AtomicBoolean UNTRUSTED_PEER_WARNED = new AtomicBoolean();

    private TrustedProxies() {}

    /** Parse a comma-separated setting. An empty value trusts no proxy. */
    public static Set<String> parse(String value) {
        Set<String> entries = new LinkedHashSet<>();
        if (value == null) return entries;
        for (String part : value.split(",")) {
            String entry = part.trim();
            if (!entry.isEmpty()) entries.add(entry);
        }
        return entries;
    }

    /** Read a config value: a JSON list or a comma-separated string. */
    public static Set<String> fromConfigValue(Object value) {
        if (value instanceof Collection<?> items) {
            Set<String> entries = new LinkedHashSet<>();
            for (Object item : items) {
                if (item != null && !String.valueOf(item).isBlank()) entries.add(String.valueOf(item).trim());
            }
            return entries;
        }
        return parse(value == null ? null : String.valueOf(value));
    }

    /** True for an IP address, a CIDR range, or one of the names "default" and "cloudflare". */
    public static boolean isValidEntry(String entry) {
        if (entry == null) return false;
        String trimmed = entry.trim();
        if (PRESETS.containsKey(trimmed.toLowerCase(Locale.ROOT))) return true;
        String[] parts = trimmed.split("/", -1);
        if (parts.length > 2 || !CidrUtil.isLiteral(parts[0])) return false;
        if (parts.length == 1) return true;
        String bits = parts[1];
        if (bits.isEmpty() || bits.length() > 3 || !bits.chars().allMatch(c -> c >= '0' && c <= '9')) return false;
        return Integer.parseInt(bits) <= (parts[0].contains(":") ? 128 : 32);
    }

    /** Replace shortcut names with the ranges they stand for. */
    public static List<String> expand(Collection<String> entries) {
        List<String> out = new ArrayList<>();
        if (entries == null) return out;
        for (String entry : entries) {
            if (entry == null) continue;
            String trimmed = entry.trim();
            out.addAll(PRESETS.getOrDefault(trimmed.toLowerCase(Locale.ROOT), List.of(trimmed)));
        }
        return out;
    }

    public static boolean contains(Collection<String> entries, String ip) {
        return compile(entries).contains(ip);
    }

    private static final Map<Set<String>, Compiled> COMPILED = new ConcurrentHashMap<>();

    /** The trusted set parsed into masks, cached by its contents so a request does no parsing. */
    public static Compiled compile(Collection<String> entries) {
        if (entries == null || entries.isEmpty()) return Compiled.EMPTY;
        Set<String> key = entries instanceof Set<String> set ? set : new LinkedHashSet<>(entries);
        Compiled compiled = COMPILED.get(key);
        if (compiled == null) {
            if (COMPILED.size() >= 64) COMPILED.clear();
            compiled = new Compiled(expand(key));
            // Keyed by a copy, so later changes to the caller's set can't corrupt the cache.
            COMPILED.put(Set.copyOf(key.stream().filter(java.util.Objects::nonNull).toList()), compiled);
        }
        return compiled;
    }

    /** Trusted networks as integer masks: IPv4 as ints, IPv6 as two longs. */
    public static final class Compiled {
        static final Compiled EMPTY = new Compiled(List.of());

        private final int[] v4Network;
        private final int[] v4Mask;
        private final long[] v6Network;
        private final long[] v6Mask;
        /** Entries that aren't a range, matched as exact text as before. */
        private final Set<String> literal;

        private Compiled(List<String> entries) {
            List<int[]> v4 = new ArrayList<>();
            List<long[]> v6 = new ArrayList<>();
            Set<String> literal = new LinkedHashSet<>();
            for (String entry : entries) {
                String[] parts = entry.split("/", -1);
                byte[] address = parts.length <= 2 ? CidrUtil.toBytes(parts[0]) : null;
                int max = address == null ? 0 : address.length * 8;
                int bits = parts.length == 1 ? max : prefix(parts.length == 2 ? parts[1] : "", max);
                if (address == null || bits < 0) {
                    literal.add(entry);
                } else if (address.length == 4) {
                    int mask = bits == 0 ? 0 : -1 << (32 - bits);
                    v4.add(new int[] {toInt(address) & mask, mask});
                } else {
                    long hiMask = bits == 0 ? 0 : bits >= 64 ? -1L : -1L << (64 - bits);
                    long loMask = bits <= 64 ? 0 : bits == 128 ? -1L : -1L << (128 - bits);
                    v6.add(new long[] {toLong(address, 0) & hiMask, toLong(address, 8) & loMask, hiMask, loMask});
                }
            }
            this.v4Network = v4.stream().mapToInt(n -> n[0]).toArray();
            this.v4Mask = v4.stream().mapToInt(n -> n[1]).toArray();
            this.v6Network = v6.stream().flatMapToLong(n -> java.util.stream.LongStream.of(n[0], n[1])).toArray();
            this.v6Mask = v6.stream().flatMapToLong(n -> java.util.stream.LongStream.of(n[2], n[3])).toArray();
            this.literal = Set.copyOf(literal);
        }

        public boolean isEmpty() {
            return v4Network.length == 0 && v6Network.length == 0 && literal.isEmpty();
        }

        public boolean contains(String ip) {
            if (ip == null) return false;
            if (!literal.isEmpty() && literal.contains(ip)) return true;
            byte[] address = CidrUtil.toBytes(ip);
            if (address == null) return false;
            if (address.length == 4) {
                int value = toInt(address);
                for (int i = 0; i < v4Network.length; i++) {
                    if ((value & v4Mask[i]) == v4Network[i]) return true;
                }
                return false;
            }
            long hi = toLong(address, 0);
            long lo = toLong(address, 8);
            for (int i = 0; i < v6Network.length; i += 2) {
                if ((hi & v6Mask[i]) == v6Network[i] && (lo & v6Mask[i + 1]) == v6Network[i + 1]) return true;
            }
            return false;
        }

        private static int prefix(String text, int max) {
            if (text.isEmpty() || !text.chars().allMatch(c -> c >= '0' && c <= '9') || text.length() > 3) return -1;
            int bits = Integer.parseInt(text);
            return bits <= max ? bits : -1;
        }

        private static int toInt(byte[] b) {
            return (b[0] & 0xff) << 24 | (b[1] & 0xff) << 16 | (b[2] & 0xff) << 8 | (b[3] & 0xff);
        }

        private static long toLong(byte[] b, int offset) {
            long value = 0;
            for (int i = offset; i < offset + 8; i++) value = value << 8 | (b[i] & 0xff);
            return value;
        }
    }

    /** Point out, once per process, a proxy that may need adding to the trusted proxies. */
    static void warnUntrustedPeerOnce(String peer, String header) {
        if (!UNTRUSTED_PEER_WARNED.compareAndSet(false, true)) return;
        LOGGER.warning("AIWAF is ignoring " + header + " from untrusted peer " + peer
                + " and using the peer's address as the client. If " + peer + " is your proxy or CDN,"
                + " add it to AIWAF_TRUSTED_PROXIES (behind Cloudflare, use"
                + " AIWAF_TRUSTED_PROXIES=default,cloudflare); otherwise no action is needed."
                + " This warning is logged once.");
    }

    static void resetUntrustedPeerWarning() {
        UNTRUSTED_PEER_WARNED.set(false);
    }
}
