package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import org.apache.catalina.Context;
import org.apache.catalina.Wrapper;
import org.apache.catalina.startup.Tomcat;
import org.apache.tomcat.websocket.server.WsSci;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.context.support.AnnotationConfigWebApplicationContext;
import org.springframework.web.servlet.DispatcherServlet;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;
import org.springframework.web.socket.handler.AbstractWebSocketHandler;

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

/** AiwafWebSocketHandler in a real Spring WebSocket server on Tomcat, with the JDK's WebSocket client. */
class AiwafWebSocketHandlerServerTest {
    private static Tomcat tomcat;
    private static int port;
    static final List<String> SEEN = new CopyOnWriteArrayList<>();

    /** Echoes text, and records binary messages. */
    static final class Echo extends AbstractWebSocketHandler {
        @Override
        protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
            SEEN.add(message.getPayload());
            session.sendMessage(new TextMessage("got " + message.getPayload()));
        }

        @Override
        protected void handleBinaryMessage(WebSocketSession session, BinaryMessage message) {
            SEEN.add("bytes");
        }
    }

    @Configuration
    @EnableWebSocket
    static class Sockets implements WebSocketConfigurer {
        @Bean
        AiwafConfig aiwafConfig() {
            return new AiwafConfig();
        }

        @Override
        public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
            registry.addHandler(new AiwafWebSocketHandler(new Echo(), aiwafConfig()), "/ws");
        }
    }

    @BeforeAll
    static void start() throws Exception {
        Path base = Files.createTempDirectory("aiwaf-spring-ws-");
        tomcat = new Tomcat();
        tomcat.setBaseDir(base.toString());
        tomcat.setPort(0);
        Context context = tomcat.addContext("", base.toString());
        context.addServletContainerInitializer(new WsSci(), null);
        AnnotationConfigWebApplicationContext spring = new AnnotationConfigWebApplicationContext();
        spring.register(Sockets.class);
        Wrapper dispatcher = Tomcat.addServlet(context, "dispatcher", new DispatcherServlet(spring));
        dispatcher.setLoadOnStartup(1);
        dispatcher.setAsyncSupported(true);
        context.addServletMappingDecoded("/", "dispatcher");
        tomcat.getConnector();
        tomcat.start();
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

    private static WebSocket connect(Client client) throws Exception {
        return HttpClient.newHttpClient().newWebSocketBuilder()
                .buildAsync(URI.create("ws://127.0.0.1:" + port + "/ws"), client).get(10, TimeUnit.SECONDS);
    }

    @Test
    void textReachesTheHandlerUntilAnAttackClosesTheSession() throws Exception {
        SEEN.clear();
        Client client = new Client();
        WebSocket socket = connect(client);
        socket.sendText("hello there", true).get(10, TimeUnit.SECONDS);
        client.replied.get(10, TimeUnit.SECONDS);
        socket.sendText("x;id", true).get(10, TimeUnit.SECONDS);
        assertClosedForPolicy(client.closed.get(10, TimeUnit.SECONDS));
        assertEquals(List.of("hello there"), SEEN);
        assertEquals(List.of("got hello there"), client.replies);
    }

    @Test
    void aBinarySerializedObjectClosesTheSession() throws Exception {
        SEEN.clear();
        Client client = new Client();
        WebSocket socket = connect(client);
        socket.sendBinary(ByteBuffer.wrap(new byte[] {(byte) 0xac, (byte) 0xed, 0, 5, 's', 'r'}), true).get(10, TimeUnit.SECONDS);
        assertClosedForPolicy(client.closed.get(10, TimeUnit.SECONDS));
        assertEquals(List.of(), SEEN);
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
