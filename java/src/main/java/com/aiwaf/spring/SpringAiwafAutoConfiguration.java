package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafEngine;
import com.aiwaf.core.UuidPolicyCore;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.core.env.Environment;
import org.springframework.web.servlet.DispatcherServlet;

/** Zero-configuration Spring Boot servlet integration, enabled unless {@code aiwaf.enabled=false}. */
@AutoConfiguration
@ConditionalOnWebApplication(type = ConditionalOnWebApplication.Type.SERVLET)
@ConditionalOnClass({DispatcherServlet.class, AiwafFilter.class})
@ConditionalOnProperty(prefix = "aiwaf", name = "enabled", matchIfMissing = true)
public class SpringAiwafAutoConfiguration {
    @Bean
    @ConditionalOnMissingBean
    public AiwafConfig aiwafConfig(Environment environment) {
        return SpringAiwafConfig.fromEnvironment(environment);
    }

    @Bean
    @ConditionalOnMissingBean
    public AiwafEngine aiwafEngine(AiwafConfig config, ApplicationContext context) {
        UuidPolicyCore.UUIDLookup lookup = SpringUuidModelLookup.discover(context);
        return new AiwafEngine(config, lookup);
    }

    @Bean
    @ConditionalOnMissingBean
    public AiwafFilter aiwafFilter(AiwafEngine engine, ApplicationContext context) {
        return new AiwafFilter(engine, context);
    }
}
