package com.aiwaf.runtime;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExemptionStoreParityTest {

    @Test
    void exact_wildcard_and_cidr_exemptions_match_python_logic() {
        ExemptionStore store = new ExemptionStore(new MemoryStorage());

        store.addIp("10.0.0.1", "exact");
        store.addPattern("10.0.*.*", "wildcard");
        store.addPattern("192.168.1.0/24", "cidr");

        assertTrue(store.isExempted("10.0.0.1"));
        assertTrue(store.isExempted("10.0.44.99"));
        assertTrue(store.isExempted("192.168.1.88"));
        assertFalse(store.isExempted("192.168.2.88"));
    }

    @Test
    void invalid_cidr_patterns_are_ignored_without_throwing() {
        ExemptionStore store = new ExemptionStore(new MemoryStorage());
        store.addPattern("not-a-cidr/abc", "invalid");
        assertFalse(store.isExempted("203.0.113.7"));
    }
}
