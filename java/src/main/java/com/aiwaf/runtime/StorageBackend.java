package com.aiwaf.runtime;

import java.util.List;

public interface StorageBackend {
    Object get(String key);
    boolean set(String key, Object value, Integer ttlSeconds);
    boolean delete(String key);
    boolean exists(String key);
    List<String> getAllKeys(String globPattern);
}
