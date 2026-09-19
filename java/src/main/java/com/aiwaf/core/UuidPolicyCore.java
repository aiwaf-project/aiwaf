package com.aiwaf.core;

import java.util.UUID;

public final class UuidPolicyCore {
    private UuidPolicyCore() {}

    public static UUIDTamperDecision evaluateUuidTamper(String uuidValue, UUIDLookup lookup) {
        if (uuidValue == null || uuidValue.isBlank()) {
            return new UUIDTamperDecision(true, null);
        }
        try {
            UUID.fromString(uuidValue);
        } catch (IllegalArgumentException ex) {
            return new UUIDTamperDecision(false, "invalid_uuid_format");
        }
        if (lookup == null) {
            return new UUIDTamperDecision(true, null);
        }
        if (lookup.exists(uuidValue)) {
            return new UUIDTamperDecision(true, null);
        }
        return new UUIDTamperDecision(false, "uuid_not_found");
    }

    public interface UUIDLookup {
        boolean exists(String uuidValue);
    }

    public record UUIDTamperDecision(boolean allow, String reason) {}
}
