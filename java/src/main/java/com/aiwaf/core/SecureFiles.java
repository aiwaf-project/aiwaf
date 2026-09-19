package com.aiwaf.core;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermission;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Set;

public final class SecureFiles {
    private static final Set<PosixFilePermission> FILE_PERMISSIONS = Set.of(
            PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE);
    private static final Set<PosixFilePermission> DIRECTORY_PERMISSIONS = Set.of(
            PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE, PosixFilePermission.OWNER_EXECUTE);

    private SecureFiles() {}

    @FunctionalInterface
    public interface OutputWriter {
        void write(OutputStream output) throws IOException;
    }

    public static void writeAtomically(Path target, OutputWriter writer) throws IOException {
        rejectSymbolicLinks(target);
        Path absolute = target.toAbsolutePath().normalize();
        Path parent = absolute.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
            restrictDirectory(parent);
        }
        Path temp = Files.createTempFile(parent, "." + absolute.getFileName(), ".tmp");
        try {
            writer.write(Files.newOutputStream(temp));
            restrictFile(temp);
            try {
                Files.move(temp, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (java.nio.file.AtomicMoveNotSupportedException ex) {
                Files.move(temp, absolute, StandardCopyOption.REPLACE_EXISTING);
            }
            restrictFile(absolute);
            writeSignatureIfConfigured(absolute);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    public static boolean verifySignature(Path target) {
        String key = System.getenv("AIWAF_ARTIFACT_HMAC_KEY");
        boolean required = Boolean.parseBoolean(System.getenv("AIWAF_REQUIRE_ARTIFACT_SIGNATURE"));
        if (key == null || key.isBlank()) return !required;
        Path signature = signaturePath(target);
        if (Files.isSymbolicLink(signature) || !Files.isRegularFile(signature)) return false;
        try {
            byte[] expected = HexFormat.of().parseHex(Files.readString(signature, StandardCharsets.US_ASCII).trim());
            return MessageDigest.isEqual(expected, hmac(target, key));
        } catch (Exception ex) {
            return false;
        }
    }

    public static void rejectSymbolicLinks(Path target) throws IOException {
        Path absolute = target.toAbsolutePath().normalize();
        Path current = absolute.getRoot();
        for (Path part : absolute) {
            current = current == null ? part : current.resolve(part);
            if (Files.isSymbolicLink(current)) {
                throw new IOException("symbolic-link persistence target rejected");
            }
        }
    }

    private static void writeSignatureIfConfigured(Path target) throws IOException {
        String key = System.getenv("AIWAF_ARTIFACT_HMAC_KEY");
        if (key == null || key.isBlank()) return;
        Path signature = signaturePath(target);
        rejectSymbolicLinks(signature);
        Path temp = Files.createTempFile(signature.toAbsolutePath().getParent(), ".aiwaf-signature-", ".tmp");
        try {
            Files.writeString(temp, HexFormat.of().formatHex(hmac(target, key)), StandardCharsets.US_ASCII);
            restrictFile(temp);
            try {
                Files.move(temp, signature, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (java.nio.file.AtomicMoveNotSupportedException ex) {
                Files.move(temp, signature, StandardCopyOption.REPLACE_EXISTING);
            }
            restrictFile(signature);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    private static byte[] hmac(Path target, String key) throws IOException {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(key.getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
            try (var input = Files.newInputStream(target)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = input.read(buffer)) >= 0) mac.update(buffer, 0, read);
            }
            return mac.doFinal();
        } catch (java.security.GeneralSecurityException ex) {
            throw new IOException("unable to calculate artifact HMAC", ex);
        }
    }

    private static Path signaturePath(Path target) {
        return target.resolveSibling(target.getFileName() + ".hmac");
    }

    public static void restrictFile(Path path) {
        try {
            if (Files.exists(path)) Files.setPosixFilePermissions(path, FILE_PERMISSIONS);
        } catch (UnsupportedOperationException | java.io.IOException ignored) {
            // Windows ACLs are inherited from the containing directory.
        }
    }

    public static void restrictDirectory(Path path) {
        try {
            if (Files.exists(path)) Files.setPosixFilePermissions(path, DIRECTORY_PERMISSIONS);
        } catch (UnsupportedOperationException | java.io.IOException ignored) {
            // Windows ACLs are inherited from the containing directory.
        }
    }
}
