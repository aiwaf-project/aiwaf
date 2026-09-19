package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertTrue;

class LegitimateRouteKeywordsCoreTest {

    static class PaymentsApiController {
        public void getInvoices() {}
        public void postInvoicePayment() {}
    }

    @Test
    void extracts_tokens_from_route_hints() {
        Set<String> out = LegitimateRouteKeywordsCore.fromRouteHints(List.of(
                "/api/v1/payments/invoices",
                "accounts-profile",
                "AdminDashboard"
        ));
        assertTrue(out.contains("payments"));
        assertTrue(out.contains("invoices"));
        assertTrue(out.contains("accounts"));
        assertTrue(out.contains("dashboard"));
    }

    @Test
    void extracts_tokens_from_handler_classes() {
        Set<String> out = LegitimateRouteKeywordsCore.fromHandlerClasses(List.of(PaymentsApiController.class));
        assertTrue(out.contains("payments"));
        assertTrue(out.contains("controller"));
        assertTrue(out.contains("invoices"));
        assertTrue(out.contains("invoice"));
    }
}
