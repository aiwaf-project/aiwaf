package com.aiwaf.servlet;

import jakarta.servlet.DispatcherType;
import jakarta.servlet.Filter;
import jakarta.servlet.MultipartConfigElement;
import org.eclipse.jetty.ee11.servlet.FilterHolder;
import org.eclipse.jetty.ee11.servlet.ServletContextHandler;
import org.eclipse.jetty.ee11.servlet.ServletHolder;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;

import java.nio.file.Path;
import java.util.EnumSet;

/** The multipart checks in an embedded Jetty 12 (Servlet 6.1). */
class JettyMultipartTest extends MultipartContainerChecks {
    private Server server;

    @Override
    int start(Path base, Filter filter) throws Exception {
        server = new Server();
        ServerConnector connector = new ServerConnector(server);
        connector.setHost("127.0.0.1");
        connector.setPort(0);
        server.addConnector(connector);
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath("/");
        ServletHolder upload = new ServletHolder(new Upload());
        upload.getRegistration().setMultipartConfig(new MultipartConfigElement(base.toString()));
        context.addServlet(upload, "/upload");
        context.addServlet(new ServletHolder(new Raw()), "/raw");
        context.addFilter(new FilterHolder(filter), "/*", EnumSet.of(DispatcherType.REQUEST));
        server.setHandler(context);
        server.start();
        return connector.getLocalPort();
    }

    @Override
    void stop() throws Exception {
        server.stop();
    }
}
