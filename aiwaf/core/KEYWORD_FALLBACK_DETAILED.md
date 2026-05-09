# Keyword Fallback Store: Detailed Technical Documentation

## 1. Scope
This document explains every relevant aspect of `aiwaf/core/keyword_fallback.py`, including:
- Its exact runtime behavior.
- How and where it is used.
- Data model and persistence semantics.
- Edge cases, failure modes, and operational risks.
- Interaction with Django keyword storage paths.
- Practical recommendations for production hardening.

Primary module:
- `aiwaf/core/keyword_fallback.py`

Primary integration point:
- `aiwaf/django/storage.py` (`ModelKeywordStore` fallback path)

## 2. Why This Module Exists
`KeywordFallbackStore` is a simple file-backed counter for learned/malicious keywords.

It exists as a resilience mechanism when Django model-backed storage is not available, specifically in `ModelKeywordStore`:
- When `DynamicKeyword` model cannot be imported/initialized.
- When a database operation on `DynamicKeyword` fails.

In these cases, keyword updates or reads fall back to JSON persistence at:
- `aiwaf/django/fallback_keywords.json`

This allows AIWAF to keep learning/tracking keywords instead of fully losing behavior during model/DB unavailability.

## 3. Source Code Walkthrough
Implementation summary (`aiwaf/core/keyword_fallback.py`):
- Maintains an in-memory `defaultdict(int)`.
- Loads whole JSON file on each operation (`add`, `top`).
- Saves whole JSON file after `add`.

### 3.1 Class: `KeywordFallbackStore`
Constructor:
- Input: `storage_path: str`
- Internal state:
  - `self.storage_path`
  - `self._keywords = defaultdict(int)`

Behavioral note:
- No immediate file read at construction.
- Data is loaded lazily per operation.

### 3.2 Method: `_load()`
Behavior:
- Checks whether `self.storage_path` exists.
- If yes, opens in read mode and `json.load(...)`.
- Rebuilds internal state as `defaultdict(int, data)`.

Implications:
- Missing file is treated as empty state.
- Entire file must be valid JSON object mapping keyword -> count.
- No schema validation beyond what `defaultdict(int, data)` tolerates.

### 3.3 Method: `_save()`
Behavior:
- Opens path in write mode.
- Serializes complete dictionary with `json.dump(..., indent=2)`.

Implications:
- Full rewrite each time `add()` is called.
- Output is human-readable JSON.
- No atomic temp-file + rename strategy.

### 3.4 Method: `add(keyword, count=1)`
Behavior:
1. Calls `_load()`.
2. Increments `self._keywords[keyword] += count`.
3. Calls `_save()`.

Implications:
- Last-write-wins under concurrent writers.
- Read-modify-write race possible across processes/threads.
- Negative `count` is accepted (can reduce totals).
- Non-string keyword would still be accepted by Python dict, but not necessarily desirable.

### 3.5 Method: `top(n=10)`
Behavior:
1. Calls `_load()`.
2. Sorts all keyword-count pairs descending by count.
3. Returns top `n` list of tuples: `[(keyword, count), ...]`.

Implications:
- Full sort cost each call.
- If counts tie, Python sort keeps insertion-order-relative behavior from loaded dict order.

## 4. Exact Data Contract
On disk, the fallback file is JSON object:

```json
{
  "keyword_a": 5,
  "keyword_b": 2
}
```

Key expectations:
- Keys should be strings.
- Values should be integers.

What happens with unexpected values:
- If value is non-numeric, arithmetic in `add()` can fail with `TypeError`.
- If JSON is invalid, `json.load` raises and caller path will fail unless upstream catches it.

## 5. Integration in Django Storage Layer
File: `aiwaf/django/storage.py`

Global initialization:
- `_fallback_storage_path = .../fallback_keywords.json`
- `_fallback_store = KeywordFallbackStore(_fallback_storage_path)`

### 5.1 `ModelKeywordStore.add_keyword`
Normal path:
- Uses `DynamicKeyword` ORM table.

Fallback path:
- If model unavailable: `_fallback_store.add(keyword, count)`.
- If DB error: logs and falls back to `_fallback_store.add(keyword, count)`.

### 5.2 `ModelKeywordStore.get_top_keywords`
Normal path:
- `DynamicKeyword.objects.order_by('-count')...`

Fallback path:
- If model unavailable or DB error: `_fallback_store.top(n)` and extract keyword names.

### 5.3 What Is Not Fallback-Backed
`ModelKeywordStore.get_all_keywords`:
- Returns `[]` when model unavailable.
- Does not read from fallback JSON.

This creates an asymmetry:
- `add_keyword` and `get_top_keywords` can use fallback.
- `get_all_keywords` does not.

Operational consequence:
- Some code paths may appear to lose keywords if they rely on `get_all_keywords` while model layer is unavailable.

## 6. Runtime Flow Scenarios

### 6.1 Healthy DB / Models Ready
- `get_keyword_store()` returns `ModelKeywordStore`.
- Reads/writes happen in DB (`DynamicKeyword`).
- Fallback JSON is not used.

### 6.2 Models Unavailable During App Lifecycle
- `add_keyword` writes to fallback JSON.
- `get_top_keywords` reads fallback JSON.
- `get_all_keywords` returns empty list.

### 6.3 Transient DB Failure
- `add_keyword` catches exception and appends count to fallback JSON.
- `get_top_keywords` catches exception and reads fallback JSON.

Note:
- There is no sync-back mechanism from fallback JSON to DB once DB recovers.
- Learned counts can diverge between fallback and DB over time.

## 7. Concurrency and Consistency Characteristics
Current implementation provides no locking.

### 7.1 Race Pattern
Two writers can do:
1. Process A loads `{x:1}`.
2. Process B loads `{x:1}`.
3. A writes `{x:2}`.
4. B writes `{x:2}`.

Expected aggregate should be 3, actual can be 2.

### 7.2 Corruption Risk Window
If process crashes during write, file can become truncated/invalid JSON.
Subsequent `_load()` may fail for all readers.

### 7.3 Cross-Platform Behavior
No file-lock strategy is used here, unlike CSV helpers in `aiwaf/core/storage_csv.py` that implement lock/retry patterns.

## 8. Performance Characteristics
Time complexity:
- `add`: O(K) to load + O(1) increment + O(K) save.
- `top(n)`: O(K) load + O(K log K) sort.

Space complexity:
- Full map held in memory each call.

Practical impact:
- Fine for small/medium keyword cardinality.
- Can become expensive with large keyword sets or high-frequency writes.

## 9. Error Handling Model
Local module (`keyword_fallback.py`) does not catch exceptions from:
- File open/read/write.
- JSON parse/serialize.
- Arithmetic type issues.

Whether failures are tolerated depends on caller:
- `ModelKeywordStore` catches DB exceptions, but fallback failures inside `_fallback_store.add/top` are not wrapped in their own protection there.

## 10. Security and Data Hygiene
This store is not an untrusted-input security boundary.

Potential concerns:
- Keyword strings are persisted verbatim; very large or unusual strings may bloat file.
- No normalization/canonicalization at fallback layer (case, whitespace, Unicode normalization).
- File path is fixed by module-level initialization in Django storage.

Recommended controls upstream:
- Normalize keywords before storage.
- Clamp keyword length.
- Clamp max count increment.

## 11. Testing Context and Coverage Notes
Observed test coverage mainly validates Django keyword behavior and persistence through model-backed store paths.

Important gap candidates for this exact fallback class:
- Corrupted JSON recovery behavior.
- Concurrent writer safety.
- Type validation for counts/values.
- Atomic write guarantees.

## 12. Operational Guidance

### 12.1 When to Inspect `fallback_keywords.json`
Inspect when:
- You see logs about DB keyword operations failing.
- Django app registry/model import instability appears.
- Top keyword behavior looks inconsistent with DB state.

### 12.2 What to Look For
- Invalid JSON structure.
- Unexpectedly large growth.
- Counts diverging from DB keyword table.

### 12.3 Recovery Steps (Current Design)
- Backup the file.
- Repair JSON if malformed.
- If desired, manually rehydrate DB from file (no built-in sync job currently).

## 13. Design Strengths
- Very small, easy-to-understand implementation.
- Zero external dependencies.
- Allows degraded operation during DB/model issues.
- Human-readable persisted format.

## 14. Design Limitations
- No atomic writes.
- No locking.
- Full file load/save each mutation.
- No schema validation.
- No automatic reconciliation with DB once healthy.
- API asymmetry (`top` fallback exists, `get_all_keywords` fallback does not).

## 15. Hardening Recommendations (Prioritized)
1. Add atomic writes (`write temp` -> `fsync` -> `rename`).
2. Add process-safe locking (platform-aware, similar to CSV store lock strategy).
3. Add robust load behavior for corrupt JSON (backup + reset strategy).
4. Validate/normalize input (`keyword`, `count`).
5. Implement optional DB re-sync from fallback store.
6. Expose metrics/log counters for fallback activation frequency.
7. Consider using shared runtime storage abstraction to unify locking/retry behavior across storage types.

## 16. Minimal API Reference

### Constructor
- `KeywordFallbackStore(storage_path: str)`

### Methods
- `_load()`:
  - Reads JSON file into in-memory keyword map if file exists.
- `_save()`:
  - Writes in-memory map to JSON file.
- `add(keyword: str, count: int = 1)`:
  - Increments keyword count and persists.
- `top(n: int = 10) -> list[tuple[str, int]]`:
  - Returns top keywords by descending count.

## 17. Compatibility Notes
- Works in Python environments without Django, since it uses only stdlib.
- In this repository, active usage is via Django storage fallback path.
- Core module is exported by `aiwaf/core/__init__.py`, making it accessible for shared usage.

## 18. Summary
`KeywordFallbackStore` is a lightweight resilience component: simple and effective for degraded-mode continuity, but intentionally minimal and currently weak on concurrency safety, corruption recovery, and lifecycle reconciliation with primary storage.

If this component is expected to handle frequent writes or multi-worker production traffic, hardening in Section 15 should be treated as mandatory rather than optional.
