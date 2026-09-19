package com.aiwaf.runtime;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RuntimeStorageBackendParityTest {

    @TempDir
    Path tempDir;

    @Test
    void file_backend_persists_exemptions_and_path_exemptions_across_reinitialize() {
        String dataFile = tempDir.resolve("aiwaf-file-store.bin").toString();

        RuntimeStorage.initialize("file", dataFile);
        RuntimeStorage.getExemptionStore().addIp("192.0.2.10", "trusted");
        RuntimeStorage.getExemptionStore().addPattern("198.51.100.*", "ops");
        RuntimeStorage.getPathExemptionStore().addPath("/health", "health check");

        RuntimeStorage.initialize("file", dataFile);
        assertTrue(RuntimeStorage.getExemptionStore().isExempted("192.0.2.10"));
        assertTrue(RuntimeStorage.getExemptionStore().isExempted("198.51.100.44"));
        assertTrue(RuntimeStorage.getPathExemptionStore().isExempted("/health", true, true));
    }

    @Test
    void csv_backend_path_exemptions_roundtrip_matches_python_flow() {
        String dataFile = tempDir.resolve("aiwaf-csv-store.bin").toString();

        RuntimeStorage.initialize("csv", dataFile);
        assertTrue(RuntimeStorage.getPathExemptionStore().addPath("/health", "Health check"));
        assertTrue(RuntimeStorage.getPathExemptionStore().addPath("/api/status", "Status"));

        assertTrue(RuntimeStorage.getPathExemptionStore().isExempted("/health", true, true));
        assertTrue(RuntimeStorage.getPathExemptionStore().isExempted("/api/status", true, true));

        assertTrue(RuntimeStorage.getPathExemptionStore().removePath("/health"));
        assertFalse(RuntimeStorage.getPathExemptionStore().isExempted("/health", true, true));

        RuntimeStorage.initialize("csv", dataFile);
        assertFalse(RuntimeStorage.getPathExemptionStore().isExempted("/health", true, true));
        assertTrue(RuntimeStorage.getPathExemptionStore().isExempted("/api/status", true, true));
    }

    @Test
    void csv_backend_serializes_complex_values_like_python_csv_storage() {
        String dataFile = tempDir.resolve("aiwaf-data.csv").toString();
        CsvStorage storage = new CsvStorage(dataFile);
        Map<String, Object> value = new HashMap<>();
        value.put("ips", List.of("192.0.2.10"));
        value.put("patterns", List.of("198.51.100.*"));

        storage.set("exemptions", value, null);

        CsvStorage reloaded = new CsvStorage(dataFile);
        Object out = reloaded.get("exemptions");
        assertInstanceOf(Map.class, out);
        Map<?, ?> map = (Map<?, ?>) out;
        assertEquals(List.of("192.0.2.10"), map.get("ips"));
        assertEquals(List.of("198.51.100.*"), map.get("patterns"));
    }

    @Test
    void csv_backend_ttl_expiry_removes_key() throws Exception {
        String dataFile = tempDir.resolve("aiwaf-data-ttl.csv").toString();
        CsvStorage storage = new CsvStorage(dataFile);
        storage.set("blocked:203.0.113.9", Map.of("reason", "ttl"), 1);
        assertTrue(storage.exists("blocked:203.0.113.9"));
        Thread.sleep(1200L);
        assertNull(storage.get("blocked:203.0.113.9"));
        assertFalse(storage.exists("blocked:203.0.113.9"));
    }
}
