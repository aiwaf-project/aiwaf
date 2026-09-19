package com.aiwaf.core;

import java.util.List;
import java.util.Map;

public record PathManifestEntry(
        String path,
        List<String> methods,
        ApiDetection apiDetection,
        Map<String, Object> authDetection
) {}
