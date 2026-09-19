package com.aiwaf.spring;

import com.aiwaf.spring.annotations.AiwafExempt;
import com.aiwaf.spring.annotations.AiwafExemptFrom;
import com.aiwaf.spring.annotations.AiwafOnly;
import com.aiwaf.spring.annotations.AiwafRequireProtection;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/")
public class TestController {
    @GetMapping("path-a")
    public ResponseEntity<String> pathA() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("path-b")
    public ResponseEntity<String> pathB() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("blocked")
    public ResponseEntity<String> blocked() {
        return ResponseEntity.ok("{\"status\":\"blocked\"}");
    }

    @GetMapping("exempted")
    public ResponseEntity<String> exempted() {
        return ResponseEntity.ok("{\"status\":\"exempted\"}");
    }

    @GetMapping("form")
    public ResponseEntity<String> formGet() {
        return ResponseEntity.ok("OK");
    }

    @PostMapping("form")
    public ResponseEntity<String> formPost() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("uuid")
    public ResponseEntity<String> uuidRoute() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("safe")
    public ResponseEntity<String> safe() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("headers")
    public ResponseEntity<String> headers() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("myapp/api/data")
    public ResponseEntity<String> apiData() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("myapp/ui")
    public ResponseEntity<String> ui() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("ann/exempt")
    @AiwafExempt
    public ResponseEntity<String> annotationExempt() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("ann/partial")
    @AiwafExemptFrom({"rate_limit"})
    public ResponseEntity<String> annotationPartial() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("ann/only")
    @AiwafOnly({"rate_limit"})
    public ResponseEntity<String> annotationOnly() {
        return ResponseEntity.ok("OK");
    }

    @GetMapping("ann/required")
    @AiwafExemptFrom({"rate_limit"})
    @AiwafRequireProtection({"rate_limit"})
    public ResponseEntity<String> annotationRequired() {
        return ResponseEntity.ok("OK");
    }
}
