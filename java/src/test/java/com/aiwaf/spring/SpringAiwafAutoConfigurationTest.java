package com.aiwaf.spring;

import com.aiwaf.core.AiwafConfig;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.WebApplicationContextRunner;
import org.springframework.mock.env.MockEnvironment;

import java.io.InputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SpringAiwafAutoConfigurationTest {
    private final WebApplicationContextRunner contextRunner = new WebApplicationContextRunner()
            .withConfiguration(AutoConfigurations.of(SpringAiwafAutoConfiguration.class));

    @Test
    void maps_spring_properties_to_core_configuration() {
        MockEnvironment environment = new MockEnvironment()
                .withProperty("aiwaf.rate-limit.max", "73")
                .withProperty("aiwaf.geo.enabled", "true")
                .withProperty("aiwaf.geo.blocked-countries", "us, ca")
                .withProperty("aiwaf.path-manifest.enabled", "false")
                .withProperty("aiwaf.path-manifest.path", "config/routes.json");

        AiwafConfig config = SpringAiwafConfig.fromEnvironment(environment);

        assertEquals(73, config.rateLimitMax);
        assertTrue(config.geoBlockEnabled);
        assertEquals(java.util.Set.of("US", "CA"), config.geoBlockedCountries);
        assertFalse(config.pathManifestEnabled);
        assertEquals("config/routes.json", config.pathManifestPath);
    }

    @Test
    void publishes_spring_boot_auto_configuration_metadata() throws Exception {
        try (InputStream stream = getClass().getClassLoader().getResourceAsStream(
                "META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports")) {
            assertNotNull(stream);
            assertTrue(new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8)
                    .contains(SpringAiwafAutoConfiguration.class.getName()));
        }
    }

    @Test
    void boot_auto_configuration_creates_the_default_runtime_and_can_be_disabled() {
        contextRunner.run(context -> {
            assertTrue(context.containsBean("aiwafConfig"));
            assertTrue(context.containsBean("aiwafEngine"));
            assertTrue(context.containsBean("aiwafFilter"));
        });
        contextRunner.withPropertyValues("aiwaf.enabled=false").run(context -> {
            assertFalse(context.containsBean("aiwafConfig"));
            assertFalse(context.containsBean("aiwafEngine"));
            assertFalse(context.containsBean("aiwafFilter"));
        });
    }
}
