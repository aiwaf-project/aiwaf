package com.aiwaf.core;

import java.io.Serializable;
import java.util.Map;

public record TrainedModelCore(String modelType, String version, Map<String, Object> payload) implements Serializable {}
