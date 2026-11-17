package com.aiwaf.runtime;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import static org.junit.jupiter.api.Assertions.*;

/** Storage reads a file again only when it changed, and still sees every change. */
class StorageChangeDetectionTest {
    @TempDir Path dir;

    @Test void anUnchangedFileHasTheSameStampAndAnyWriteChangesIt() throws Exception {
        Path file = dir.resolve("store.csv");
        assertNull(FileStamp.of(file));
        CsvStorage storage = new CsvStorage(file.toString());
        storage.set("a", "1", null);
        FileStamp first = FileStamp.of(file);
        assertEquals(first, FileStamp.of(file));
        storage.set("a", "2", null);
        assertNotEquals(first, FileStamp.of(file));
    }

    @Test void anOutsideEditWithTheSameSizeAndModificationTimeIsStillRead() throws Exception {
        Path file = dir.resolve("store.csv");
        CsvStorage reader = new CsvStorage(file.toString());
        new CsvStorage(file.toString()).set("blocked:1.2.3.4", "x", null);
        assertEquals("x", reader.get("blocked:1.2.3.4"));
        FileTime modified = Files.getLastModifiedTime(file);
        String edited = Files.readString(file, StandardCharsets.UTF_8).replace("x", "y");
        Thread.sleep(20);
        Files.writeString(file, edited, StandardCharsets.UTF_8);
        Files.setLastModifiedTime(file, modified);
        assertEquals("y", reader.get("blocked:1.2.3.4"));
    }

    @Test void writesFromAnotherInstanceAreSeenByBothBackends() {
        for (String type : new String[]{"csv", "file"}) {
            String file = dir.resolve(type + "-shared").toString();
            StorageBackend a = type.equals("csv") ? new CsvStorage(file) : new FileStorage(file);
            StorageBackend b = type.equals("csv") ? new CsvStorage(file) : new FileStorage(file);
            assertNull(b.get("k"));
            a.set("k", "1", null);
            assertEquals("1", b.get("k"));
            for (int i = 0; i < 3; i++) assertEquals("1", b.get("k"));
            a.delete("k");
            assertNull(b.get("k"));
            assertTrue(b.getAllKeys("*").isEmpty());
        }
    }

    @Test void aRemovedLockDirectoryIsCreatedAgain() throws Exception {
        Path nested = dir.resolve("nested");
        Files.createDirectories(nested);
        CsvStorage storage = new CsvStorage(nested.resolve("store.csv").toString());
        storage.set("k", "1", null);
        Files.delete(nested.resolve("store.csv.lock"));
        Files.delete(nested.resolve("store.csv"));
        Files.delete(nested);
        assertNull(storage.get("k"));
        assertTrue(storage.set("k", "2", null));
        assertEquals("2", storage.get("k"));
    }
}
