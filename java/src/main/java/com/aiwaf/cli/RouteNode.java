package com.aiwaf.cli;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class RouteNode {
    private final String segment;
    private final RouteNode parent;
    private boolean endpoint;
    private final Map<String, RouteNode> children = new LinkedHashMap<>();

    public RouteNode(String segment, RouteNode parent) {
        this.segment = segment;
        this.parent = parent;
    }

    public String segment() {
        return segment;
    }

    public RouteNode parent() {
        return parent;
    }

    public boolean endpoint() {
        return endpoint;
    }

    public void setEndpoint(boolean endpoint) {
        this.endpoint = endpoint;
    }

    public Map<String, RouteNode> children() {
        return children;
    }

    public RouteNode child(String name) {
        return children.get(name);
    }

    public RouteNode childAt(int indexOneBased) {
        if (indexOneBased <= 0) return null;
        List<RouteNode> values = new ArrayList<>(children.values());
        int idx = indexOneBased - 1;
        if (idx >= values.size()) return null;
        return values.get(idx);
    }
}
