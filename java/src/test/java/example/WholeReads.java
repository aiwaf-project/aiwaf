package example;

import jakarta.servlet.http.HttpServletRequest;
import java.io.IOException;

/** Application code, outside the framework packages, reading the request whole. */
public final class WholeReads {
    private WholeReads() {}

    public static void parameters(HttpServletRequest request) {
        request.getParameterMap();
    }

    public static void names(HttpServletRequest request) {
        request.getParameterNames();
    }

    public static void stream(HttpServletRequest request) throws IOException {
        request.getInputStream();
    }

    public static void reader(HttpServletRequest request) throws IOException {
        request.getReader();
    }
}
