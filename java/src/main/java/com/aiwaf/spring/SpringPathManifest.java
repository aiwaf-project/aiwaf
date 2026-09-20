package com.aiwaf.spring;

import com.aiwaf.core.PathManifestCore;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.method.HandlerMethod;
import org.springframework.web.servlet.mvc.method.RequestMappingInfo;
import org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerMapping;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/** Live Spring application route discovery for the Java equivalent of {@code aiwaf init}. */
public final class SpringPathManifest {
    private SpringPathManifest() {}

    public static List<PathManifestCore.RouteInfo> discoverRoutes(ApplicationContext context) {
        if (context == null) throw new IllegalArgumentException("application context must not be null");
        List<PathManifestCore.RouteInfo> routes = new ArrayList<>();
        Map<String, RequestMappingHandlerMapping> mappings = context.getBeansOfType(RequestMappingHandlerMapping.class);
        for (RequestMappingHandlerMapping mapping : mappings.values()) {
            for (Map.Entry<RequestMappingInfo, HandlerMethod> entry : mapping.getHandlerMethods().entrySet()) {
                RequestMappingInfo info = entry.getKey();
                HandlerMethod handler = entry.getValue();
                List<String> methods = info.getMethodsCondition().getMethods().stream()
                        .map(RequestMethod::name)
                        .sorted()
                        .toList();
                if (methods.isEmpty()) methods = List.of("DELETE", "GET", "PATCH", "POST", "PUT");
                for (String path : info.getPatternValues()) {
                    routes.add(new PathManifestCore.RouteInfo(
                            PathManifestCore.normalizePath(path),
                            methods,
                            handler.getBeanType(),
                            handler.getMethod()
                    ));
                }
            }
        }
        routes.sort(Comparator.comparing(PathManifestCore.RouteInfo::path)
                .thenComparing(route -> route.method().getName()));
        return routes;
    }

    public static Map<String, Object> generate(ApplicationContext context, String outputPath) {
        List<PathManifestCore.RouteInfo> routes = discoverRoutes(context);
        Map<String, Object> manifest = PathManifestCore.buildManifest("spring", routes);
        PathManifestCore.writeManifest(manifest, Path.of(
                outputPath == null || outputPath.isBlank() ? PathManifestCore.DEFAULT_MANIFEST_PATH : outputPath));
        return manifest;
    }

    /** Start, inspect, and close a Spring Boot or plain annotation application context. */
    public static Map<String, Object> launchAndGenerate(String applicationClassName, String outputPath, String... applicationArgs) {
        if (applicationClassName == null || applicationClassName.isBlank()) {
            throw new IllegalArgumentException("--app requires a Spring application class");
        }
        ConfigurableApplicationContext context = null;
        try {
            Class<?> applicationClass = Class.forName(applicationClassName);
            context = launch(applicationClass, applicationArgs == null ? new String[0] : applicationArgs);
            return generate(context, outputPath);
        } catch (ReflectiveOperationException ex) {
            Throwable cause = ex instanceof InvocationTargetException invocation && invocation.getCause() != null
                    ? invocation.getCause() : ex;
            throw new IllegalStateException("Could not initialize Spring application " + applicationClassName
                    + ": " + cause.getMessage(), cause);
        } finally {
            if (context != null) context.close();
        }
    }

    private static ConfigurableApplicationContext launch(Class<?> applicationClass, String[] applicationArgs)
            throws ReflectiveOperationException {
        try {
            Class<?> springApplication = Class.forName("org.springframework.boot.SpringApplication");
            Method run = springApplication.getMethod("run", Class.class, String[].class);
            Object context = run.invoke(null, applicationClass, applicationArgs);
            if (!(context instanceof ConfigurableApplicationContext configurable)) {
                throw new IllegalStateException("SpringApplication.run did not return an application context");
            }
            return configurable;
        } catch (ClassNotFoundException ignored) {
            AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext();
            context.register(applicationClass);
            context.refresh();
            return context;
        }
    }
}
