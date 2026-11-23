package com.aiwaf.servlet;

import jakarta.servlet.http.HttpServletRequest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Recording which parameters handlers read, from a servlet filter. */
class AiwafInputRecorderFilterTest {
    @TempDir
    Path directory;

    @Test
    void destinationFollowsTheSetting() {
        assertNull(AiwafInputRecorderFilter.destination(null));
        assertNull(AiwafInputRecorderFilter.destination("off"));
        assertEquals(Path.of(AiwafInputRecorderFilter.DEFAULT_PATH), AiwafInputRecorderFilter.destination("1"));
        assertEquals(Path.of("x.jsonl"), AiwafInputRecorderFilter.destination("x.jsonl"));
    }

    @Test
    void recordsWhatTheHandlerReadsByItsRoutePattern() throws Exception {
        Path target = directory.resolve("out/seen.jsonl");
        AiwafInputRecorderFilter filter = new AiwafInputRecorderFilter(target);
        filter.init(null);
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/items/1");
        request.setParameter("q", "a");
        request.setParameter("extra", "b");
        request.setAttribute("org.springframework.web.servlet.HandlerMapping.bestMatchingPattern", "/items/{id}");
        filter.doFilter(request, new MockHttpServletResponse(), new MockFilterChain(new jakarta.servlet.http.HttpServlet() {
            @Override
            protected void service(HttpServletRequest req, jakarta.servlet.http.HttpServletResponse res) {
                req.getParameter("q");
                req.getParameterValues("page");
                req.getParameter("_csrf");
                req.getParameterMap();
            }
        }));
        MockHttpServletRequest servlet = new MockHttpServletRequest("POST", "/plain");
        servlet.setServletPath("/plain");
        servlet.setPathInfo("/more");
        filter.doFilter(servlet, new MockHttpServletResponse(), new MockFilterChain(new jakarta.servlet.http.HttpServlet() {
            @Override
            protected void service(HttpServletRequest req, jakarta.servlet.http.HttpServletResponse res) {
                req.getParameter("name");
            }
        }));
        filter.doFilter(new MockHttpServletRequest("GET", "/none"), new MockHttpServletResponse(), new MockFilterChain());
        List<String> lines = Files.readAllLines(target);
        assertEquals(List.of(
                "{\"route\":\"/items/{id}\",\"method\":\"GET\",\"key\":\"page\"}",
                "{\"route\":\"/items/{id}\",\"method\":\"GET\",\"key\":\"q\"}",
                "{\"route\":\"/plain/more\",\"method\":\"POST\",\"key\":\"name\"}"), lines);
    }

    @Test
    void offAndUnwritableOnlyPassTheRequestOn() throws Exception {
        MockFilterChain chain = new MockFilterChain();
        new AiwafInputRecorderFilter((Path) null).doFilter(new MockHttpServletRequest(), new MockHttpServletResponse(), chain);
        assertTrue(chain.getRequest() != null);
        new AiwafInputRecorderFilter().doFilter(new MockHttpServletRequest(), new MockHttpServletResponse(), new MockFilterChain());
        Path blocked = Files.writeString(directory.resolve("file"), "");
        AiwafInputRecorderFilter filter = new AiwafInputRecorderFilter(blocked.resolve("seen.jsonl"));
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/q");
        request.setServletPath("/q");
        filter.doFilter(request, new MockHttpServletResponse(), new MockFilterChain(new jakarta.servlet.http.HttpServlet() {
            @Override
            protected void service(HttpServletRequest req, jakarta.servlet.http.HttpServletResponse res) {
                req.getParameter("q");
            }
        }));
        assertFalse(Files.exists(blocked.resolve("seen.jsonl")));
    }
}
