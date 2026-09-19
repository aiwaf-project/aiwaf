package com.aiwaf.core;

public final class BlockPolicyCore {
    private BlockPolicyCore() {}

    public static boolean canBlockIp(boolean isExempt, BlockPolicyConfig config) {
        if (config == null || !config.enableIpBlocking()) {
            return false;
        }
        return !isExempt;
    }

    public record BlockPolicyConfig(boolean enableIpBlocking) {
        public BlockPolicyConfig() {
            this(true);
        }
    }
}
