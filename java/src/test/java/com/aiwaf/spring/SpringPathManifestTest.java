package com.aiwaf.spring;

import com.aiwaf.core.PathManifestCore;
import org.junit.jupiter.api.Test;
import org.springframework.context.support.StaticApplicationContext;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.RequestMappingInfo;
import org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerMapping;

import java.lang.reflect.Method;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SpringPathManifestTest {
    @RestController
    static class LiveController {
        @GetMapping("/live/{uuid}")
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
        context.close();
    }
}
