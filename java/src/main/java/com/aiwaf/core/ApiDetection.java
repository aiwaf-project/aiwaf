package com.aiwaf.core;

import java.util.List;

public record ApiDetection(
        boolean isApi,
        String responseType,
        String payloadType,
        double confidence,
        List<String> signals,
        boolean requestBody,
        double formConfidence,
        List<String> formSignals
) {}
