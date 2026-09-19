package com.aiwaf.core;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputFilter;
import java.io.ObjectInputStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Creates object streams with an allowlist and resource limits installed before
 * any object can be instantiated.
 */
public final class SafeObjectInputStreams {
    public enum Profile { MODEL, CONFIG, STORAGE }
    private static final long MAX_DEPTH = 64;
    private static final long MAX_REFERENCES = 100_000;
    private static final long MAX_ARRAY_LENGTH = 1_000_000;
    private static final long MAX_STREAM_BYTES = 64L * 1024L * 1024L;

    private SafeObjectInputStreams() {}

    public static ObjectInputStream open(InputStream input) throws IOException {
        return open(input, Profile.CONFIG);
    }

    public static ObjectInputStream open(InputStream input, Profile profile) throws IOException {
        ObjectInputStream stream = new ObjectInputStream(input);
        stream.setObjectInputFilter(info -> check(info, profile));
        return stream;
    }

    private static ObjectInputFilter.Status check(ObjectInputFilter.FilterInfo info, Profile profile) {
        if (info.depth() > MAX_DEPTH
                || info.references() > MAX_REFERENCES
                || info.arrayLength() > MAX_ARRAY_LENGTH
                || info.streamBytes() > MAX_STREAM_BYTES) {
            return ObjectInputFilter.Status.REJECTED;
        }

        Class<?> type = info.serialClass();
        if (type == null) {
            return ObjectInputFilter.Status.UNDECIDED;
        }
        while (type.isArray()) {
            type = type.getComponentType();
        }
        if (type.isPrimitive() || isScalar(type) || isAllowedCollection(type) || isProfileClass(type, profile)) {
            return ObjectInputFilter.Status.ALLOWED;
        }
        return ObjectInputFilter.Status.REJECTED;
    }

    private static boolean isProfileClass(Class<?> type, Profile profile) {
        String name = type.getName();
        return switch (profile) {
            case MODEL -> name.equals("com.aiwaf.core.TrainedModelCore")
                    || name.startsWith("com.aiwaf.core.IsolationForestCore$");
            case STORAGE -> name.equals("com.aiwaf.runtime.FileStorage$Entry");
            case CONFIG -> false;
        };
    }

    private static boolean isScalar(Class<?> type) {
        return type == Object.class
                || type == String.class
                || type == Number.class
                || type == Boolean.class
                || type == Byte.class
                || type == Short.class
                || type == Integer.class
                || type == Long.class
                || type == Float.class
                || type == Double.class
                || type == Character.class
                || type == BigInteger.class
                || type == BigDecimal.class
                || type.isEnum()
                || type.getName().startsWith("java.time.");
    }

    private static boolean isAllowedCollection(Class<?> type) {
        String name = type.getName();
        return type == ArrayList.class
                || type == LinkedList.class
                || type == HashMap.class
                || type == LinkedHashMap.class
                || type == TreeMap.class
                || type == ConcurrentHashMap.class
                || type == HashSet.class
                || type == LinkedHashSet.class
                || type == TreeSet.class
                || type == Map.Entry.class
                || name.equals("java.util.CollSer")
                || name.equals("java.util.Arrays$ArrayList")
                || name.equals("java.util.concurrent.ConcurrentHashMap$Segment")
                || name.startsWith("java.util.ImmutableCollections$");
    }
}
