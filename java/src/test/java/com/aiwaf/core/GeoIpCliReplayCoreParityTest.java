package com.aiwaf.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;

class GeoIpCliReplayCoreParityTest {

    @TempDir
    Path tempDir;

    @Test
    void lookup_country_missing_db_returns_none() {
        Path missing = tempDir.resolve("missing.mmdb");
        assertNull(GeoIpCore.lookupCountry("1.1.1.1", missing.toString()));
    }

    @Test
    void default_mmdb_path_is_packaged_path() {
        String mmdb = GeoIpCore.defaultMmdbPath();
        assertTrue(mmdb.endsWith("geolock/ipinfo_lite.mmdb") || mmdb.endsWith("geolock\\ipinfo_lite.mmdb"));
    }

    @Test
    void lookup_country_name_missing_db_returns_none() {
        Path missing = tempDir.resolve("missing.mmdb");
        assertNull(GeoIpCore.lookupCountryName("1.1.1.1", missing.toString()));
    }

    @Test
    void mmdblookup_availability_gate_is_safe_when_tool_missing() {
        boolean available = GeoIpCore.isMmdbLookupAvailable();
        String mmdb = GeoIpCore.defaultMmdbPath();
        if (!available) {
            assertNull(GeoIpCore.lookupCountry("8.8.8.8", mmdb));
            assertNull(GeoIpCore.lookupCountryName("8.8.8.8", mmdb));
        } else {
            assertFalse(mmdb.isBlank());
        }
    }

    @Test
    void lookup_country_cached_returns_cached_value() {
        Path missing = tempDir.resolve("missing.mmdb");
        Map<String, String> seen = new HashMap<>();
        seen.put("geo:1.1.1.1", "US");

        String out = GeoIpCore.lookupCountryCached(
                "1.1.1.1",
                missing.toString(),
                "geo:1.1.1.1",
                60,
                seen::get,
                call -> seen.put(call.key(), call.value())
        );
        assertEquals("US", out);
    }

    @Test
    void lookup_country_cached_stores_null_when_lookup_misses() {
        Path missing = tempDir.resolve("missing.mmdb");
        Map<String, String> cache = new HashMap<>();
        List<GeoIpCore.CacheSetCall> writes = new java.util.ArrayList<>();

        String out = GeoIpCore.lookupCountryCached(
                "1.1.1.1",
                missing.toString(),
                "geo:1.1.1.1",
                60,
                cache::get,
                writes::add
        );

        assertNull(out);
        assertEquals(1, writes.size());
        assertEquals("geo:1.1.1.1", writes.get(0).key());
        assertNull(writes.get(0).value());
        assertEquals(60, writes.get(0).timeoutSeconds());
    }

    @Test
    void extract_country_from_raw_parity_paths() {
        Map<String, Object> rawCountryCode = new HashMap<>();
        rawCountryCode.put("country_code", "US");
        assertEquals("US", GeoIpCore.extractCountryFromRaw(rawCountryCode));

        Map<String, Object> nestedCountry = new HashMap<>();
        nestedCountry.put("country", Map.of("iso_code", "FR"));
        assertEquals("FR", GeoIpCore.extractCountryFromRaw(nestedCountry));

        Map<String, Object> countryString = new HashMap<>();
        countryString.put("country", "Canada");
        assertEquals("Canada", GeoIpCore.extractCountryFromRaw(countryString));
    }

    @Test
    void extract_country_name_from_raw_parity_paths() {
        Map<String, Object> nestedCountry = new HashMap<>();
        nestedCountry.put("country", Map.of("name", "United States"));
        assertEquals("United States", GeoIpCore.extractCountryNameFromRaw(nestedCountry));

        Map<String, Object> countryString = new HashMap<>();
        countryString.put("country", "Canada");
        assertEquals("Canada", GeoIpCore.extractCountryNameFromRaw(countryString));

        Map<String, Object> countryName = new HashMap<>();
        countryName.put("country_name", "France");
        assertEquals("France", GeoIpCore.extractCountryNameFromRaw(countryName));
    }

    @Test
    void replay_loader_parses_cases() throws Exception {
        var resource = GeoIpCliReplayCoreParityTest.class.getClassLoader().getResource("replay_cases.json");
        assertNotNull(resource);
        List<ReplayCore.ReplayCase> cases = ReplayCore.loadReplayCases(Path.of(resource.toURI()));
        assertTrue(cases.size() >= 2);
        assertTrue(cases.get(0).name() != null && !cases.get(0).name().isBlank());
    }

    @Test
    void cli_analyze_behavior_and_reset() throws Exception {
        String eventsPath = Path.of(GeoIpCliReplayCoreParityTest.class.getClassLoader().getResource("events.json").toURI()).toString();
        int rc1 = CoreCli.main(new String[]{"analyze-behavior", "--events", eventsPath});
        int rc2 = CoreCli.main(new String[]{"reset", "--targets", "all"});
        assertEquals(0, rc1);
        assertEquals(0, rc2);
    }

    @Test
    void cli_list_and_reset_validate_targets() {
        int listOk = CoreCli.main(new String[]{"list", "--targets", "features,models"});
        int resetOk = CoreCli.main(new String[]{"reset", "--targets", "events"});
        int listInvalid = CoreCli.main(new String[]{"list", "--targets", "unknown"});
        int resetInvalid = CoreCli.main(new String[]{"reset", "--targets", "all,features"});
        assertEquals(0, listOk);
        assertEquals(0, resetOk);
        assertEquals(2, listInvalid);
        assertEquals(2, resetInvalid);
    }

    @Test
    void cli_analyze_request() throws Exception {
        Path reqJson = tempDir.resolve("req.json");
        Files.writeString(
                reqJson,
                "{\"ip\":\"4.4.4.4\",\"method\":\"GET\",\"path\":\"/home\",\"headers\":{\"user-agent\":\"Mozilla/5.0 Chrome/120\",\"accept\":\"text/html,application/xml\",\"accept-language\":\"en-US\",\"accept-encoding\":\"gzip\",\"connection\":\"keep-alive\"},\"known_path\":true}",
                StandardCharsets.UTF_8
        );
        int rc = CoreCli.main(new String[]{"analyze-request", "--request", reqJson.toString()});
        assertEquals(0, rc);
    }
}
