package com.aiwaf.core;

public record AiwafDecision(boolean allowed, int statusCode, String reason) {
    public static AiwafDecision allow() {
        return new AiwafDecision(true, 200, "OK");
    }

    public static AiwafDecision deny(int statusCode, String reason) {
        return new AiwafDecision(false, statusCode, reason);
    }
}
