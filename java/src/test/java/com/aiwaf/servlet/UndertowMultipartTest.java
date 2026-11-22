package com.aiwaf.servlet;

import io.undertow.Undertow;
import io.undertow.UndertowOptions;
import io.undertow.servlet.Servlets;
import io.undertow.servlet.api.DeploymentInfo;
import io.undertow.servlet.api.DeploymentManager;
import io.undertow.servlet.util.ImmediateInstanceFactory;
import jakarta.servlet.DispatcherType;
import jakarta.servlet.Filter;
import jakarta.servlet.MultipartConfigElement;
import jakarta.servlet.Servlet;

import java.net.InetSocketAddress;
import java.nio.file.Path;

/** The multipart checks in an embedded Undertow 2.3 (Servlet 6.0). */
class UndertowMultipartTest extends MultipartContainerChecks {
    private Undertow server;
    private DeploymentManager manager;

    @Override
    int start(Path base, Filter filter) throws Exception {
        DeploymentInfo deployment = Servlets.deployment()
                .setClassLoader(UndertowMultipartTest.class.getClassLoader())
                .setContextPath("/")
                .setDeploymentName("aiwaf-multipart")
                .addServlet(Servlets.servlet("upload", Upload.class, new ImmediateInstanceFactory<Servlet>(new Upload()))
                        .addMapping("/upload")
                        .setMultipartConfig(new MultipartConfigElement(base.toString())))
                .addServlet(Servlets.servlet("raw", Raw.class, new ImmediateInstanceFactory<Servlet>(new Raw())).addMapping("/raw"))
                .addFilter(Servlets.filter("aiwaf", AiwafServletFilter.class, new ImmediateInstanceFactory<Filter>(filter)))
                .addFilterUrlMapping("aiwaf", "/*", DispatcherType.REQUEST);
        manager = Servlets.defaultContainer().addDeployment(deployment);
        manager.deploy();
        // Undertow caps request bodies itself, so it's raised to the size the checks send.
        server = Undertow.builder().addHttpListener(0, "127.0.0.1")
                .setServerOption(UndertowOptions.MAX_ENTITY_SIZE, 16L * 1024 * 1024)
                .setServerOption(UndertowOptions.MULTIPART_MAX_ENTITY_SIZE, 16L * 1024 * 1024)
                .setHandler(manager.start()).build();
        server.start();
        return ((InetSocketAddress) server.getListenerInfo().get(0).getAddress()).getPort();
    }

    @Override
    void stop() throws Exception {
        server.stop();
        manager.stop();
        manager.undeploy();
    }
}
