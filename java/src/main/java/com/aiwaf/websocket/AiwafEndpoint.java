package com.aiwaf.websocket;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafMessageCore;
import jakarta.websocket.CloseReason;
import jakarta.websocket.Endpoint;
import jakarta.websocket.EndpointConfig;
import jakarta.websocket.MessageHandler;
import jakarta.websocket.Session;

import java.io.IOException;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Proxy;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;

/**
 * A Jakarta WebSocket endpoint that checks each message a client sends before the wrapped
 * endpoint's handlers see it, as request bodies are checked. Register it in place of the
 * endpoint, as in a {@code ServerEndpointConfig.Configurator} returning
 * {@code new AiwafEndpoint(new ChatEndpoint(), config)}. A message with a finding closes the
 * session with 1008 (1009 when too large to check). It follows {@code sqlInjectionMode}.
 *
 * <p>The wrapped endpoint's message handlers are checked: text ({@code String}) and binary
 * ({@code ByteBuffer} or {@code byte[]}), whole or in parts, each part on its own. A handler
 * added with {@code addMessageHandler(handler)} must name its message type in its class, as a
 * class implementing {@code MessageHandler.Whole<String>}; a lambda can't be read, so add it
 * with {@code addMessageHandler(String.class, handler)}. For annotated {@code @ServerEndpoint}
 * classes, call {@link AiwafMessageCore#check} in the {@code @OnMessage} method.
 */
public class AiwafEndpoint extends Endpoint {
    private final Endpoint delegate;
    private final AiwafConfig config;

    public AiwafEndpoint(Endpoint delegate, AiwafConfig config) {
        this.delegate = delegate;
        this.config = config;
    }

    @Override
    public void onOpen(Session session, EndpointConfig endpointConfig) {
        delegate.onOpen(checkedSession(session), endpointConfig);
    }

    @Override
    public void onClose(Session session, CloseReason closeReason) {
        delegate.onClose(session, closeReason);
    }

    @Override
    public void onError(Session session, Throwable error) {
        delegate.onError(session, error);
    }

    /** True to let the message through; otherwise the session is closed. */
    boolean allowed(Session session, Object message) {
        Object payload = message;
        if (message instanceof ByteBuffer buffer) {
            ByteBuffer copy = buffer.asReadOnlyBuffer();
            byte[] bytes = new byte[copy.remaining()];
            copy.get(bytes);
            payload = bytes;
        }
        String path = session.getRequestURI() == null ? "" : session.getRequestURI().getPath();
        String rule = AiwafMessageCore.check(payload, config, path);
        if (rule == null) return true;
        try {
            session.close(new CloseReason(CloseReason.CloseCodes.getCloseCode(AiwafMessageCore.closeCode(rule)), "aiwaf"));
        } catch (IOException ignored) {
            // The client is gone already.
        }
        return false;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private MessageHandler checked(Session session, MessageHandler handler) {
        if (handler instanceof MessageHandler.Partial partial) {
            return (MessageHandler.Partial<Object>) (part, last) -> {
                if (allowed(session, part)) partial.onMessage(part, last);
            };
        }
        MessageHandler.Whole whole = (MessageHandler.Whole) handler;
        return (MessageHandler.Whole<Object>) message -> {
            if (allowed(session, message)) whole.onMessage(message);
        };
    }

    /** The message type a handler class names, as T in MessageHandler.Whole<T>, or null. */
    static Class<?> messageType(MessageHandler handler) {
        for (Class<?> type = handler.getClass(); type != null; type = type.getSuperclass()) {
            for (Type implemented : type.getGenericInterfaces()) {
                if (implemented instanceof ParameterizedType parameterized
                        && (parameterized.getRawType() == MessageHandler.Whole.class || parameterized.getRawType() == MessageHandler.Partial.class)
                        && parameterized.getActualTypeArguments()[0] instanceof Class<?> argument) {
                    return argument;
                }
            }
        }
        return null;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private Session checkedSession(Session session) {
        InvocationHandler calls = (proxy, method, args) -> {
            if (method.getName().equals("addMessageHandler")) {
                if (args.length == 1) {
                    MessageHandler handler = (MessageHandler) args[0];
                    Class<?> type = messageType(handler);
                    if (type == null) {
                        throw new IllegalArgumentException("AiwafEndpoint can't read this handler's message type; "
                                + "add it with addMessageHandler(String.class, handler) or a ByteBuffer.class form");
                    }
                    return register(session, type, checked(session, handler));
                }
                return register(session, (Class<?>) args[0], checked(session, (MessageHandler) args[1]));
            }
            try {
                return method.invoke(session, args);
            } catch (InvocationTargetException ex) {
                throw ex.getCause();
            }
        };
        return (Session) Proxy.newProxyInstance(Session.class.getClassLoader(), new Class<?>[] {Session.class}, calls);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static Object register(Session session, Class type, MessageHandler handler) {
        if (handler instanceof MessageHandler.Partial partial) {
            session.addMessageHandler(type, partial);
        } else {
            session.addMessageHandler(type, (MessageHandler.Whole) handler);
        }
        return null;
    }
}
