package com.aiwaf.runtime;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;

/**
 * What identifies a storage file's current contents: its inode, change time, modification time
 * and size. Saves replace the file atomically, so each gets a new inode; the change time can't
 * be set back by writers, unlike the modification time. Equal stamps mean an unchanged file.
 */
record FileStamp(Object fileKey, long changed, long modified, long size) {
    /** The stamp of {@code file}, or null when it doesn't exist. */
    static FileStamp of(Path file) {
        try {
            BasicFileAttributes attributes = Files.readAttributes(file, BasicFileAttributes.class, LinkOption.NOFOLLOW_LINKS);
            long modified = attributes.lastModifiedTime().to(java.util.concurrent.TimeUnit.NANOSECONDS);
            return new FileStamp(attributes.fileKey(), changeTime(file, modified), modified, attributes.size());
        } catch (IOException ex) {
            return null;
        }
    }

    private static long changeTime(Path file, long fallback) {
        try {
            Object value = Files.getAttribute(file, "unix:ctime", LinkOption.NOFOLLOW_LINKS);
            return value instanceof FileTime time ? time.to(java.util.concurrent.TimeUnit.NANOSECONDS) : fallback;
        } catch (UnsupportedOperationException | IllegalArgumentException | IOException ex) {
            return fallback;  // Not a Unix file system: the modification time alone.
        }
    }
}
