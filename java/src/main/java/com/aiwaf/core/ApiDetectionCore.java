package com.aiwaf.core;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ApiDetectionCore {

    public static ApiDetection detectApiEndpoint(Method method, Class<?> controllerClass, String path) {
        int score = 0;
        int formScore = 0;
        List<String> signals = new ArrayList<>();
        List<String> formSignals = new ArrayList<>();
        boolean requestBody = false;
        String payloadType = "";

        String pathLower = path != null ? path.toLowerCase(Locale.ROOT) : "";

        if (pathLower.startsWith("/api/") || pathLower.startsWith("/v1/") || pathLower.startsWith("/v2/") || pathLower.contains("/api/")) {
            score += 30;
            signals.add("path:/api");
        }
        if (pathLower.endsWith("/graphql") || pathLower.endsWith("/graphql/") || pathLower.endsWith("/token") || pathLower.endsWith("/token/")) {
            score += 30;
            signals.add("path:graphql_or_token");
        }
        if (pathLower.contains("/webhook") || pathLower.contains("/callback")) {
            score += 25;
            signals.add("path:webhook");
        }

        if (controllerClass != null) {
            if (hasAnnotation(controllerClass, "org.springframework.web.bind.annotation.RestController")) {
                score += 50;
                signals.add("class:@RestController");
                payloadType = "json";
            }
            if (hasAnnotation(controllerClass, "org.springframework.stereotype.Controller") && !hasAnnotation(controllerClass, "org.springframework.web.bind.annotation.RestController")) {
                formScore += 30;
                formSignals.add("class:@Controller");
            }
        }

        if (method != null) {
            if (hasAnnotation(method, "org.springframework.web.bind.annotation.ResponseBody")) {
                score += 40;
                signals.add("method:@ResponseBody");
                payloadType = "json";
            }

            // Check return type
            Class<?> returnType = method.getReturnType();
            if (returnType.getName().equals("org.springframework.http.ResponseEntity") ||
                returnType.getName().equals("java.util.Map") ||
                returnType.getName().equals("java.util.List")) {
                score += 40;
                signals.add("return_annotation:" + returnType.getSimpleName());
            } else if (returnType.getName().equals("java.lang.String") ||
                       returnType.getName().equals("org.springframework.web.servlet.ModelAndView")) {
                if (formScore > 0) { // Only count as form if it's a @Controller
                    formScore += 40;
                    formSignals.add("return_annotation:" + returnType.getSimpleName());
                }
            }

            // Check parameters
            for (java.lang.reflect.Parameter param : method.getParameters()) {
                if (hasAnnotation(param, "org.springframework.web.bind.annotation.RequestBody")) {
                    score += 50;
                    signals.add("param:@RequestBody");
                    requestBody = true;
                    payloadType = "json";
                }
                if (hasAnnotation(param, "org.springframework.web.bind.annotation.ModelAttribute")) {
                    formScore += 45;
                    formSignals.add("param:@ModelAttribute");
                    requestBody = true;
                    payloadType = "form";
                }
            }
        }

        boolean isForm = formScore >= 50 && formScore >= score;
        boolean isApi = score >= 50 && !isForm;

        double confidence = Math.min(0.99, score / 100.0);
        double formConfidence = Math.min(0.99, formScore / 100.0);
        String responseType = "";

        if (isApi) {
            responseType = "json";
            if (payloadType.isEmpty()) payloadType = "json";
        } else if (isForm) {
            responseType = score > 0 ? "mixed" : "html";
            payloadType = "form";
        }

        return new ApiDetection(isApi, responseType, payloadType, confidence, signals, requestBody, formConfidence, formSignals);
    }

    private static boolean hasAnnotation(java.lang.reflect.AnnotatedElement element, String annotationName) {
        for (java.lang.annotation.Annotation a : element.getAnnotations()) {
            if (a.annotationType().getName().equals(annotationName)) {
                return true;
            }
        }
        return false;
    }
}
