package com.aiwaf.core;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.mock.env.MockEnvironment;
import org.springframework.mock.web.MockHttpServletRequest;

import com.aiwaf.spring.SpringAiwafConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrustedProxiesTest {
    private static final String PUBLIC_PEER = "198.51.100.7";
    private static final String PROXY = "127.0.0.1";
    private static final String CLIENT = "203.0.113.5";
    private static final String CLOUDFLARE_PEER = "104.16.0.1";
    private static final String CLOUDFLARE_PEER_V6 = "2606:4700::1";

    private final Logger logger = Logger.getLogger(TrustedProxies.class.getName());
    private final List<String> warnings = new ArrayList<>();
    private final Handler capture = new Handler() {
        @Override public void publish(LogRecord record) { warnings.add(record.getMessage()); }
        @Override public void flush() {}
        @Override public void close() {}
    };

    @BeforeEach
    void captureWarnings() {
        TrustedProxies.resetUntrustedPeerWarning();
        logger.addHandler(capture);
    }

    @AfterEach
    void stopCapturing() {
        logger.removeHandler(capture);
        TrustedProxies.resetUntrustedPeerWarning();
    }

    private static String resolve(String peer, String forwardedFor, Set<String> trusted) {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.setRemoteAddr(peer);
        if (forwardedFor != null) request.addHeader("X-Forwarded-For", forwardedFor);
        AiwafConfig config = new AiwafConfig();
        if (trusted != null) config.trustedProxyCidrs = trusted;
        return ServletRequestMapper.from(request, Set.of(), config).ip();
    }

    private List<String> proxyWarnings() {
        return warnings.stream().filter(m -> m.contains("untrusted peer")).toList();
    }

    @ParameterizedTest
    @ValueSource(strings = {"cloudflare", "Cloudflare"})
    void cloudflare_shortcut_trusts_cloudflare_ranges_only(String name) {
        Set<String> trusted = TrustedProxies.parse(name);
        assertEquals(CLIENT, resolve(CLOUDFLARE_PEER, CLIENT, trusted));
        assertEquals(CLIENT, resolve(CLOUDFLARE_PEER_V6, CLIENT, trusted));
        // Setting the list replaces the defaults, so a local proxy is no longer trusted.
        assertEquals(PROXY, resolve(PROXY, CLIENT, trusted));
        assertEquals(PUBLIC_PEER, resolve(PUBLIC_PEER, CLIENT, trusted));
    }

    @Test
    void default_shortcut_keeps_builtin_ranges_alongside_cloudflare() {
        Set<String> trusted = TrustedProxies.parse("default, cloudflare");
        assertEquals(CLIENT, resolve(PROXY, CLIENT, trusted));
        assertEquals(CLIENT, resolve(CLOUDFLARE_PEER, CLIENT, trusted));
        assertEquals(CLIENT, resolve(PROXY, CLIENT + ", " + CLOUDFLARE_PEER, trusted));
        assertEquals(PUBLIC_PEER, resolve(PUBLIC_PEER, CLIENT, trusted));
    }

    @Test
    void environment_variable_and_spring_property_replace_the_default() {
        AiwafConfig fromEnv = new AiwafConfig();
        AiwafConfigCompatCore.applyEnv(fromEnv, Map.of("AIWAF_TRUSTED_PROXIES", "default,cloudflare"));
        assertEquals(Set.of("default", "cloudflare"), fromEnv.trustedProxyCidrs);

        AiwafConfig trustNone = new AiwafConfig();
        AiwafConfigCompatCore.applyEnv(trustNone, Map.of("AIWAF_TRUSTED_PROXIES", ""));
        assertTrue(trustNone.trustedProxyCidrs.isEmpty());

        AiwafConfig fromSpring = SpringAiwafConfig.fromEnvironment(
                new MockEnvironment().withProperty("aiwaf.trusted-proxies", "cloudflare, 10.0.0.0/8"));
        assertEquals(Set.of("cloudflare", "10.0.0.0/8"), fromSpring.trustedProxyCidrs);
        assertEquals(Set.copyOf(TrustedProxies.DEFAULT), new AiwafConfig().trustedProxyCidrs);
    }

    @Test
    void untrusted_peer_sending_forwarded_for_is_reported_once() {
        assertEquals(PUBLIC_PEER, resolve(PUBLIC_PEER, CLIENT, null));
        assertEquals("198.51.100.8", resolve("198.51.100.8", CLIENT, null));
        List<String> messages = proxyWarnings();
        assertEquals(1, messages.size());
        assertTrue(messages.get(0).contains("X-Forwarded-For"));
        assertTrue(messages.get(0).contains(PUBLIC_PEER));
        assertTrue(messages.get(0).contains("default,cloudflare"));
    }

    @Test
    void untrusted_peer_sending_only_real_ip_is_reported() {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.setRemoteAddr(PUBLIC_PEER);
        request.addHeader("X-Real-IP", CLIENT);
        assertEquals(PUBLIC_PEER, ServletRequestMapper.from(request, Set.of(), new AiwafConfig()).ip());
        assertTrue(proxyWarnings().get(0).contains("X-Real-IP"));
    }

    @Test
    void no_warning_without_a_misconfigured_proxy() {
        resolve(PUBLIC_PEER, null, null);           // a direct client with no proxy headers
        resolve(PROXY, CLIENT, null);               // a trusted proxy
        resolve(PUBLIC_PEER, CLIENT, Set.of());     // trusting no proxy was chosen deliberately
        assertEquals(List.of(), proxyWarnings());
    }

    private static Path writeConfig(Path dir, String trustedProxiesJson) throws IOException {
        Path file = dir.resolve("aiwaf.json");
        Files.writeString(file, "{\"security\": {\"trusted_proxies\": " + trustedProxiesJson + "}}");
        return file;
    }

    @Test
    void json_config_accepts_a_list_or_a_comma_separated_string(@TempDir Path dir) throws IOException {
        AiwafConfig fromList = AiwafConfigFileCore.load(writeConfig(dir, "[\"default\", \"cloudflare\"]"), Map.of());
        assertEquals(Set.of("default", "cloudflare"), fromList.trustedProxyCidrs);
        AiwafConfig fromString = AiwafConfigFileCore.load(writeConfig(dir, "\"10.0.0.0/8, cloudflare\""), Map.of());
        assertEquals(Set.of("10.0.0.0/8", "cloudflare"), fromString.trustedProxyCidrs);
    }

    @Test
    void json_config_empty_list_trusts_no_proxy_and_null_keeps_the_default(@TempDir Path dir) throws IOException {
        assertTrue(AiwafConfigFileCore.load(writeConfig(dir, "[]"), Map.of()).trustedProxyCidrs.isEmpty());
        assertEquals(Set.copyOf(TrustedProxies.DEFAULT),
                AiwafConfigFileCore.load(writeConfig(dir, "null"), Map.of()).trustedProxyCidrs);
    }

    @Test
    void environment_variable_overrides_the_json_config(@TempDir Path dir) throws IOException {
        AiwafConfig config = AiwafConfigFileCore.load(
                writeConfig(dir, "[\"cloudflare\"]"), Map.of("AIWAF_TRUSTED_PROXIES", "192.0.2.1"));
        assertEquals(Set.of("192.0.2.1"), config.trustedProxyCidrs);
    }

    @Test
    void saved_config_keeps_trusted_proxies(@TempDir Path dir) throws IOException {
        AiwafConfig original = new AiwafConfig();
        original.trustedProxyCidrs = TrustedProxies.parse("default,cloudflare,2001:db8::/32");
        Path file = dir.resolve("saved.json");
        AiwafConfigFileCore.save(original, file);
        assertEquals(original.trustedProxyCidrs, AiwafConfigFileCore.load(file, Map.of()).trustedProxyCidrs);
    }

    @Test
    void invalid_entries_are_rejected_when_loading(@TempDir Path dir) throws IOException {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> AiwafConfigFileCore.load(writeConfig(dir, "[\"cloudflare\", \"10.0.0.0/99\", \"cdn\"]"), Map.of()));
        assertTrue(error.getMessage().contains("security.trusted_proxies entry 10.0.0.0/99"));
        assertTrue(error.getMessage().contains("security.trusted_proxies entry cdn"));
    }

    @Test
    void entry_validation() {
        for (String valid : List.of("default", "CLOUDFLARE", "10.0.0.1", "10.0.0.0/8", "::1", "2001:db8::/32", "0.0.0.0/0")) {
            assertTrue(TrustedProxies.isValidEntry(valid), valid);
        }
        for (String invalid : List.of("", "cdn", "10.0.0.0/33", "2001:db8::/129", "10.0.0.0/x", "10.0.0.0/+8", "10.0.0/8", "10.0.0.0/8/1", "host.example")) {
            assertFalse(TrustedProxies.isValidEntry(invalid), invalid);
        }
    }

    /** The check used before trusted sets were compiled, kept here as the reference. */
    private static boolean previousContains(Set<String> entries, String ip) {
        for (String cidr : TrustedProxies.expand(entries)) {
            if (cidr.equals(ip) || ("::1/128".equals(cidr) && "::1".equals(ip))
                    || com.aiwaf.runtime.CidrUtil.contains(cidr, ip)) {
                return true;
            }
        }
        return false;
    }

    @Test
    void compiled_set_matches_the_previous_check() {
        List<Set<String>> configs = List.of(
                Set.of("default"), Set.of("default", "cloudflare"), Set.of("cloudflare"),
                Set.of("10.0.0.0/8", "192.168.1.0/24", "2001:db8::/32", "fe80::/10", "203.0.113.9"),
                Set.of("0.0.0.0/0", "::/0"), Set.of("10.0.0.1/32", "::1/128", "2001:db8::1/128"),
                Set.of("10.0.0.0/7", "172.16.0.0/12", "2001:db8:8000::/33", "2001:db8::/65", "2001:db8::/127"),
                Set.of("unknown", "10.0.0.0/33", "bad/8"), Set.of());
        List<String> ips = List.of(
                "127.0.0.1", "127.255.255.255", "128.0.0.1", "10.0.0.1", "11.0.0.1", "9.255.255.255",
                "172.15.255.255", "172.16.0.1", "172.31.255.255", "172.32.0.0", "192.168.1.200", "192.168.2.1",
                "203.0.113.9", "203.0.113.10", "104.16.0.1", "131.0.72.1", "131.0.76.0", "::1", "::2",
                "2001:db8::1", "2001:db8::2", "2001:db8:8000::1", "2001:db8:7fff::1", "2001:db8::8000:0:0:1",
                "2001:db9::1", "fe80::1", "febf::1", "fec0::1", "2606:4700::1", "2c0f:f248::1", "unknown", "");
        for (Set<String> config : configs) {
            TrustedProxies.Compiled compiled = TrustedProxies.compile(config);
            for (String ip : ips) {
                assertEquals(previousContains(config, ip), compiled.contains(ip), config + " " + ip);
            }
        }
    }

    @Test
    void compiled_sets_follow_changes_to_the_configured_set() {
        Set<String> configured = new java.util.HashSet<>(Set.of("10.0.0.0/8"));
        assertTrue(TrustedProxies.compile(configured).contains("10.1.2.3"));
        configured.clear();
        configured.add("192.168.0.0/16");
        assertFalse(TrustedProxies.compile(configured).contains("10.1.2.3"));
        assertTrue(TrustedProxies.compile(configured).contains("192.168.4.4"));
        assertTrue(TrustedProxies.compile(Set.of()).isEmpty());
        assertTrue(TrustedProxies.compile(null).isEmpty());
    }
}
