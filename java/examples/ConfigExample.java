import com.aiwaf.core.AiwafConfig;
import com.aiwaf.core.AiwafDecision;
import com.aiwaf.core.AiwafEngine;
import com.aiwaf.core.AiwafRequest;

import java.util.Map;
import java.util.Set;

public final class ConfigExample {
    public static void main(String[] args) {
        AiwafConfig config = new AiwafConfig();
        config.headerValidationEnabled = true;
        config.rateLimitEnabled = true;
        config.rateLimitWindowSeconds = 10;
        config.rateLimitMax = 20;
        config.honeypotEnabled = true;
        config.ipKeywordBlockEnabled = true;
        config.geoBlockEnabled = false;
        config.uuidTamperEnabled = true;
        config.exemptPaths = Set.of("/health");

        AiwafEngine engine = new AiwafEngine(config);

        AiwafRequest request = new AiwafRequest(
                "GET",
                "/api/protected",
                "127.0.0.1",
                "US",
                Map.of("user-agent", "curl/8.0", "accept", "*/*", "host", "localhost"),
                Map.of(),
                System.currentTimeMillis(),
                Set.of()
        );

        long startedAt = System.currentTimeMillis();
        AiwafDecision decision = engine.evaluateBeforeResponse(request);
        if (decision.allowed()) {
            decision = engine.evaluateAfterResponse(
                    request,
                    200,
                    System.currentTimeMillis() - startedAt
            );
        }
        System.out.println("Allowed=" + decision.allowed() + ", status=" + decision.statusCode() + ", reason=" + decision.reason());
    }
}
