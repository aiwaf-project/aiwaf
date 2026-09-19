package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InvalidClassException;
import java.io.ObjectOutputStream;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SafeObjectInputStreamsTest {

    @Test
    void accepts_supported_data_shapes() throws Exception {
        Map<String, Object> value = new HashMap<>();
        value.put("count", 3);
        value.put("name", "safe");

        try (var in = SafeObjectInputStreams.open(new ByteArrayInputStream(serialize(value)))) {
            assertEquals(value, in.readObject());
        }
    }

    @Test
    void rejects_classes_outside_the_allowlist_before_readObject_completes() throws Exception {
        byte[] bytes = serialize(new URL("https://example.invalid/"));

        try (var in = SafeObjectInputStreams.open(new ByteArrayInputStream(bytes))) {
            assertThrows(InvalidClassException.class, in::readObject);
        }
    }

    private static byte[] serialize(Object value) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(value);
        }
        return bytes.toByteArray();
    }
}
