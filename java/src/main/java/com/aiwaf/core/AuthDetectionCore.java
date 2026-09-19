package com.aiwaf.core;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import java.util.Locale;

public class AuthDetectionCore {

    public static Map<String, Object> detectAuthEndpoint(String path, Method method, Class<?> controllerClass) {
        Map<String, Object> result = new HashMap<>();
        String p = path != null ? path.toLowerCase(Locale.ROOT) : "";

        boolean isAuth = false;
        String authType = null;

        if (p.contains("/login") || p.contains("/signin") || p.contains("/auth") || p.contains("/token")) {
            isAuth = true;
            authType = "login";
        } else if (p.contains("/logout") || p.contains("/signout")) {
            isAuth = true;
            authType = "logout";
        } else if (p.contains("/register") || p.contains("/signup")) {
            isAuth = true;
            authType = "register";
        }

        if (!isAuth && method != null) {
            String mName = method.getName().toLowerCase(Locale.ROOT);
            if (mName.contains("login") || mName.contains("authenticate") || mName.contains("signin")) {
                isAuth = true;
                authType = "login";
            } else if (mName.contains("logout") || mName.contains("signout")) {
                isAuth = true;
                authType = "logout";
            } else if (mName.contains("register") || mName.contains("signup")) {
                isAuth = true;
                authType = "register";
            }
        }

        result.put("is_auth", isAuth);
        if (isAuth && authType != null) {
            result.put("auth_type", authType);
        }

        return result;
    }
}
