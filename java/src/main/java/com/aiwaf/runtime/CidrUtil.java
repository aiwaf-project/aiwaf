package com.aiwaf.runtime;

import java.net.InetAddress;

public final class CidrUtil {
    private CidrUtil() {}

    public static boolean contains(String cidr, String ip) {
        try {
            String[] parts = cidr.split("/", -1);
            if (parts.length != 2) return false;
            int prefix = Integer.parseInt(parts[1]);
            byte[] network = parseLiteral(parts[0]);
            byte[] target = parseLiteral(ip);
            if (network.length != target.length || prefix < 0 || prefix > network.length * 8) return false;
            int wholeBytes = prefix / 8;
            int remainingBits = prefix % 8;
            for (int i = 0; i < wholeBytes; i++) {
                if (network[i] != target[i]) return false;
            }
            if (remainingBits == 0) return true;
            int mask = 0xff << (8 - remainingBits);
            return (network[wholeBytes] & mask) == (target[wholeBytes] & mask);
        } catch (Exception e) {
            return false;
        }
    }

    public static boolean isLiteral(String value) {
        try {
            parseLiteral(value);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    private static byte[] parseLiteral(String value) throws Exception {
        if (value == null || value.isBlank() || value.contains("%")) throw new IllegalArgumentException("invalid IP");
        if (value.contains(":")) {
            if (!value.matches("[0-9a-fA-F:.]+")) throw new IllegalArgumentException("invalid IPv6");
            byte[] bytes = InetAddress.getByName(value).getAddress();
            if (bytes.length != 16) throw new IllegalArgumentException("invalid IPv6");
            return bytes;
        }
        String[] parts = value.split("\\.", -1);
        if (parts.length != 4) throw new IllegalArgumentException("invalid IPv4");
        byte[] bytes = new byte[4];
        for (int i = 0; i < 4; i++) {
            if (!parts[i].matches("\\d{1,3}")) throw new IllegalArgumentException("invalid IPv4");
            int octet = Integer.parseInt(parts[i]);
            if (octet > 255) throw new IllegalArgumentException("invalid IPv4");
            bytes[i] = (byte) octet;
        }
        return bytes;
    }
}
