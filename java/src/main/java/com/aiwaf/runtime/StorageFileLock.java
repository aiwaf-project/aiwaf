package com.aiwaf.runtime;

import com.aiwaf.core.SecureFiles;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/** Stable sidecar locks survive atomic replacement of the storage data file. */
final class StorageFileLock {
    private static final Object[] STRIPES = new Object[64];
    static {
        for (int i = 0; i < STRIPES.length; i++) STRIPES[i] = new Object();
    }

    /** Lock directories already created, so each call doesn't create them again. */
    private static final Set<Path> PREPARED = ConcurrentHashMap.newKeySet();

    private StorageFileLock() {}

    /** An exclusive lock, for reading and writing the file. */
    static <T> T withLock(Path file, Supplier<T> action) {
        return locked(file, false, action);
    }

    /**
     * A shared lock, for only reading the file: readers in other processes don't wait for each
     * other. Within one JVM, locks on a file still take turns, as Java requires.
     */
    static <T> T withReadLock(Path file, Supplier<T> action) {
        return locked(file, true, action);
    }

    private static <T> T locked(Path file, boolean shared, Supplier<T> action) {
        Path absolute = file.toAbsolutePath().normalize();
        Path lock = absolute.resolveSibling(absolute.getFileName() + ".lock");
        // Prevent overlapping OS locks from different instances in this JVM.
        synchronized (STRIPES[Math.floorMod(lock.hashCode(), STRIPES.length)]) {
            try {
                SecureFiles.rejectSymbolicLinks(absolute);
                SecureFiles.rejectSymbolicLinks(lock);
                if (!PREPARED.contains(lock.getParent())) {
                    Files.createDirectories(lock.getParent());
                    PREPARED.add(lock.getParent());
                }
                try (FileChannel channel = open(lock);
                     var held = channel.lock(0, Long.MAX_VALUE, shared)) {
                    return action.get();
                }
            } catch (IOException ex) {
                throw new IllegalStateException("Unable to lock storage file " + absolute, ex);
            }
        }
    }

    private static FileChannel open(Path lock) throws IOException {
        try {
            return FileChannel.open(lock, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE);
        } catch (java.nio.file.NoSuchFileException ex) {
            // The directory was removed while the application ran: create it again.
            Files.createDirectories(lock.getParent());
            return FileChannel.open(lock, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE);
        }
    }
}
