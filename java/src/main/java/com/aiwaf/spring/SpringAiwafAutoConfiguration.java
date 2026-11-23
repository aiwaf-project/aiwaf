package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafEngine;
import com.aiwaf.core.UuidPolicyCore;
import com.aiwaf.servlet.AiwafInputRecorderFilter;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication;
import org.springframework.context.ApplicationContext;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Condition;
import org.springframework.context.annotation.ConditionContext;
import org.springframework.context.annotation.Conditional;
import org.springframework.core.Ordered;
import org.springframework.core.type.AnnotatedTypeMetadata;
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

    /**
     * Records what each handler reads, for init, when {@code aiwaf.record-inputs} or
     * {@code AIWAF_RECORD_INPUTS} is set (a file, or 1 for .aiwaf/observed_inputs.jsonl). It runs
     * last of the filters, so AIWAF's own reads aren't the handler's.
     */
    @Bean
    @ConditionalOnMissingBean(AiwafInputRecorderFilter.class)
    @Conditional(RecordingInputs.class)
    public FilterRegistrationBean<AiwafInputRecorderFilter> aiwafInputRecorderFilter(Environment environment) {
        FilterRegistrationBean<AiwafInputRecorderFilter> registration =
                new FilterRegistrationBean<>(new AiwafInputRecorderFilter(AiwafInputRecorderFilter.destination(RecordingInputs.setting(environment))));
        registration.setOrder(Ordered.LOWEST_PRECEDENCE);
        return registration;
    }

    static final class RecordingInputs implements Condition {
        static String setting(Environment environment) {
            return environment.getProperty("aiwaf.record-inputs", environment.getProperty(AiwafInputRecorderFilter.ENVIRONMENT));
        }

        @Override
        public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
            return AiwafInputRecorderFilter.destination(setting(context.getEnvironment())) != null;
        }
    }
}
