package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafMessageCore;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.WebSocketMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.WebSocketHandlerDecorator;

import java.nio.ByteBuffer;

/**
 * Checks each message a client sends before the wrapped handler sees it, as request bodies are
 * checked: {@code registry.addHandler(new AiwafWebSocketHandler(handler, config), "/ws")}. A
 * message with a finding closes the session with 1008 (1009 when too large to check). It follows
 * {@code sqlInjectionMode}: "monitor" logs and passes the message, "off" checks nothing.
 */
public class AiwafWebSocketHandler extends WebSocketHandlerDecorator {
    private final AiwafConfig config;

    public AiwafWebSocketHandler(WebSocketHandler delegate, AiwafConfig config) {
        super(delegate);
        this.config = config;
    }

    @Override
    public void handleMessage(WebSocketSession session, WebSocketMessage<?> message) throws Exception {
        Object payload = null;
        if (message instanceof TextMessage text) {
            payload = text.getPayload();
        } else if (message instanceof BinaryMessage binary) {
            ByteBuffer buffer = binary.getPayload().asReadOnlyBuffer();
            byte[] bytes = new byte[buffer.remaining()];
            buffer.get(bytes);
            payload = bytes;
        }
        String path = session.getUri() == null ? "" : session.getUri().getPath();
        String rule = payload == null ? null : AiwafMessageCore.check(payload, config, path);
        if (rule != null) {
            session.close(new CloseStatus(AiwafMessageCore.closeCode(rule), "aiwaf"));
            return;
        }
        super.handleMessage(session, message);
    }
}
