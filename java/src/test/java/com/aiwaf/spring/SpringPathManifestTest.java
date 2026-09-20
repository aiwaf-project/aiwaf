package com.aiwaf.spring;

import com.aiwaf.core.PathManifestCore;
import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafEngine;
import com.aiwaf.spring.annotations.AiwafExempt;
import org.junit.jupiter.api.Test;
import org.springframework.context.support.StaticApplicationContext;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.RequestMappingInfo;
import org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerMapping;

import java.lang.reflect.Method;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;

class SpringPathManifestTest {
    @RestController
    static class LiveController {
        @GetMapping("/live/{uuid}")
        @AiwafExempt
        public String live() { return "ok"; }
    }

    @Test
    void discovers_registered_request_mapping_handler_methods() throws Exception {
        StaticApplicationContext context = new StaticApplicationContext();
        RequestMappingHandlerMapping mapping = new RequestMappingHandlerMapping();
        mapping.setApplicationContext(context);
        mapping.afterPropertiesSet();
        LiveController controller = new LiveController();
        Method method = LiveController.class.getMethod("live");
        mapping.registerMapping(
                RequestMappingInfo.paths("/live/{uuid}").methods(org.springframework.web.bind.annotation.RequestMethod.GET).build(),
                controller,
                method
        );
        context.getBeanFactory().registerSingleton("requestMappingHandlerMapping", mapping);

        List<PathManifestCore.RouteInfo> routes = SpringPathManifest.discoverRoutes(context);

        assertEquals(1, routes.size());
        assertEquals("/live/{uuid}", routes.get(0).path());
        assertEquals(List.of("GET"), routes.get(0).httpMethods());
        List<SpringPathManifest.MappedRoute> mappedRoutes = SpringPathManifest.discoverHandlerRoutes(context);
        assertEquals(1, mappedRoutes.size());
        assertSame(controller, mappedRoutes.get(0).handler().getBean());

        AiwafFilter filter = new AiwafFilter(new AiwafEngine(new AiwafConfig()), context);
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/live/not-a-uuid");
        request.setRemoteAddr("203.0.113.20");
        MockHttpServletResponse response = new MockHttpServletResponse();
        MockFilterChain chain = new MockFilterChain();
        filter.doFilter(request, response, chain);
        assertNotNull(chain.getRequest());
        context.close();
    }
}
