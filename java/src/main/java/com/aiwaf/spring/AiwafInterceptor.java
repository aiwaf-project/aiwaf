package com.aiwaf.spring;

import com.aiwaf.core.AiwafDecision;
import com.aiwaf.core.AiwafEngine;
import com.aiwaf.core.AiwafRequest;
import com.aiwaf.core.ExemptionsCore;
import com.aiwaf.core.ServletRequestMapper;
import com.aiwaf.spring.support.AiwafRouteDecisions;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.web.servlet.HandlerInterceptor;

import java.util.HashSet;
import java.util.Set;

public final class AiwafInterceptor implements HandlerInterceptor {
    private static final String[] MIDDLEWARES = {
            "ip_keyword_block", "rate_limit", "honeypot", "header_validation", "geo_block", "uuid_tamper"
    };

    private final AiwafEngine engine;

    public AiwafInterceptor(AiwafEngine engine) {
        this.engine = engine;
    }

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        AiwafRequest mapped = ServletRequestMapper.from(request, java.util.Set.of(), engine.config());
        Set<String> disabled = new HashSet<>();
        for (String middleware : MIDDLEWARES) {
            if (!AiwafRouteDecisions.shouldApply(request.getRequestURI(), handler, middleware, engine.config().pathRules)) {
                disabled.add(ExemptionsCore.normalizeMiddlewareName(middleware));
            }
        }
        request.setAttribute("aiwaf.disabled.middlewares", disabled);
        AiwafDecision decision = engine.evaluate(mapped.withDisabledMiddlewares(disabled));
        if (!decision.allowed()) {
            response.sendError(decision.statusCode(), decision.reason());
            return false;
        }
        request.setAttribute("aiwaf.already.checked", Boolean.TRUE);
        return true;
    }
}
