package com.aiwaf.runtime;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DbStorage implements StorageBackend {
    private static final Logger logger = Logger.getLogger(DbStorage.class.getName());
    private static final ObjectMapper mapper = new ObjectMapper();
    private final String dbUrl;

    public DbStorage(String filePath) {
        File f = new File(filePath);
        File parent = f.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        this.dbUrl = "jdbc:sqlite:" + f.getAbsolutePath();
        initTable();
    }

    private void initTable() {
        try (Connection conn = getConnection(); Statement stmt = conn.createStatement()) {
            stmt.execute("CREATE TABLE IF NOT EXISTS kv_store (" +
                         "key TEXT PRIMARY KEY, " +
                         "value TEXT NOT NULL, " +
                         "expires_at REAL)");
        } catch (SQLException e) {
            logger.log(Level.SEVERE, "Failed to initialize DB storage table", e);
        }
    }

    private Connection getConnection() throws SQLException {
        return DriverManager.getConnection(dbUrl);
    }

    private void cleanupExpired(Connection conn) throws SQLException {
        long now = System.currentTimeMillis() / 1000;
        try (PreparedStatement stmt = conn.prepareStatement("DELETE FROM kv_store WHERE expires_at IS NOT NULL AND expires_at < ?")) {
            stmt.setDouble(1, (double) now);
            stmt.executeUpdate();
        }
    }

    @Override
    public Object get(String key) {
        try (Connection conn = getConnection()) {
            cleanupExpired(conn);
            try (PreparedStatement stmt = conn.prepareStatement("SELECT value FROM kv_store WHERE key = ?")) {
                stmt.setString(1, key);
                try (ResultSet rs = stmt.executeQuery()) {
                    if (rs.next()) {
                        String jsonValue = rs.getString(1);
                        try {
                            return mapper.readValue(jsonValue, Object.class);
                        } catch (JsonProcessingException e) {
                            return jsonValue;
                        }
                    }
                }
            }
        } catch (SQLException e) {
            logger.log(Level.SEVERE, "Failed to get key " + key, e);
        }
        return null;
    }

    @Override
    public boolean set(String key, Object value, Integer ttlSeconds) {
        try (Connection conn = getConnection()) {
            Double expiresAt = null;
            if (ttlSeconds != null) {
                expiresAt = (System.currentTimeMillis() / 1000.0) + ttlSeconds;
            }
            String payload;
            try {
                payload = mapper.writeValueAsString(value);
            } catch (JsonProcessingException e) {
                payload = value.toString();
            }

            try (PreparedStatement stmt = conn.prepareStatement("INSERT OR REPLACE INTO kv_store (key, value, expires_at) VALUES (?, ?, ?)")) {
                stmt.setString(1, key);
                stmt.setString(2, payload);
                if (expiresAt != null) {
                    stmt.setDouble(3, expiresAt);
                } else {
                    stmt.setNull(3, java.sql.Types.REAL);
                }
                stmt.executeUpdate();
                return true;
            }
        } catch (SQLException e) {
            logger.log(Level.SEVERE, "Failed to set key " + key, e);
            return false;
        }
    }

    @Override
    public boolean delete(String key) {
        try (Connection conn = getConnection()) {
            try (PreparedStatement stmt = conn.prepareStatement("DELETE FROM kv_store WHERE key = ?")) {
                stmt.setString(1, key);
                return stmt.executeUpdate() > 0;
            }
        } catch (SQLException e) {
            logger.log(Level.SEVERE, "Failed to delete key " + key, e);
            return false;
        }
    }

    @Override
    public boolean exists(String key) {
        return get(key) != null;
    }

    @Override
    public List<String> getAllKeys(String pattern) {
        List<String> keys = new ArrayList<>();
        try (Connection conn = getConnection()) {
            cleanupExpired(conn);
            try (Statement stmt = conn.createStatement(); ResultSet rs = stmt.executeQuery("SELECT key FROM kv_store")) {
                while (rs.next()) {
                    String k = rs.getString(1);
                    if ("*".equals(pattern) || k.matches(pattern.replace("*", ".*"))) {
                        keys.add(k);
                    }
                }
            }
        } catch (SQLException e) {
            logger.log(Level.SEVERE, "Failed to get all keys", e);
        }
        return keys;
    }
}
