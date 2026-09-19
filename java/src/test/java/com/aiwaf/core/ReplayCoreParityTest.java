package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.net.URI;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ReplayCoreParityTest {

    @Test
    void replay_fixture_parity() throws Exception {
        URI resource = ReplayCoreParityTest.class.getClassLoader().getResource("replay_cases.json").toURI();
        List<ReplayCore.ReplayCase> cases = ReplayCore.loadReplayCases(Path.of(resource));

        AiwafConfig config = new AiwafConfig();
        config.methodValidationEnabled = true;
        config.headerValidationEnabled = false;
        config.rateLimitEnabled = false;
        config.ipKeywordBlockEnabled = true;

        List<String> failures = ReplayCore.assertReplayParity(cases, config);
        assertEquals(List.of(), failures);
    }
}
