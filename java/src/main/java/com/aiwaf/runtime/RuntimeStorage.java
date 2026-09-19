package com.aiwaf.runtime;

public final class RuntimeStorage {
    private static StorageBackend backend;
    private static ExemptionStore exemptionStore;
    private static PathExemptionStore pathExemptionStore;
    private static BlacklistStore blacklistStore;
    private static KeywordStore keywordStore;
    private static GeoBlockStore geoBlockStore;

    private RuntimeStorage() {}

    public static synchronized StorageBackend initialize(String backendType, String filePath) {
        String kind = backendType == null ? "memory" : backendType.trim().toLowerCase();
        switch (kind) {
            case "memory" -> backend = new MemoryStorage();
            case "file" -> backend = new FileStorage(filePath == null ? "aiwaf_data.bin" : filePath);
            case "csv" -> backend = new CsvStorage(filePath == null ? "aiwaf_data.csv" : filePath);
            case "db" -> backend = new DbStorage(filePath == null ? "aiwaf_data.db" : filePath);
            default -> throw new IllegalArgumentException("Unknown storage backend: " + backendType);
        }
        exemptionStore = new ExemptionStore(backend);
        pathExemptionStore = new PathExemptionStore(backend);
        blacklistStore = new BlacklistStore(backend);
        keywordStore = new KeywordStore(backend);
        geoBlockStore = new GeoBlockStore(backend);
        return backend;
    }

    public static synchronized StorageBackend getStorage() {
        if (backend == null) initialize("memory", null);
        return backend;
    }

    public static synchronized ExemptionStore getExemptionStore() {
        if (exemptionStore == null) initialize("memory", null);
        return exemptionStore;
    }

    public static synchronized BlacklistStore getBlacklistStore() {
        if (blacklistStore == null) initialize("memory", null);
        return blacklistStore;
    }

    public static synchronized PathExemptionStore getPathExemptionStore() {
        if (pathExemptionStore == null) initialize("memory", null);
        return pathExemptionStore;
    }

    public static synchronized KeywordStore getKeywordStore() {
        if (keywordStore == null) initialize("memory", null);
        return keywordStore;
    }

    public static synchronized GeoBlockStore getGeoBlockStore() {
        if (geoBlockStore == null) initialize("memory", null);
        return geoBlockStore;
    }
}
