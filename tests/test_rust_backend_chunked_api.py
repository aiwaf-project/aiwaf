#!/usr/bin/env python3
"""Tests for chunked Rust feature extraction wrappers."""

import aiwaf.rust_backend as rb


def test_supports_chunked_feature_extraction_requires_both_symbols():
    original = rb.aiwaf_rust
    try:
        class OnlyBatch:
            def extract_features_batch_with_state(self, records, keywords, state):
                return {"features": [], "state": state}

        rb.aiwaf_rust = OnlyBatch()
        assert rb.supports_chunked_feature_extraction() is False

        class FullAPI:
            def extract_features_batch_with_state(self, records, keywords, state):
                return {"features": [], "state": state}

            def finalize_feature_state(self, keywords, state):
                return {"features": []}

        rb.aiwaf_rust = FullAPI()
        assert rb.supports_chunked_feature_extraction() is True
    finally:
        rb.aiwaf_rust = original


def test_extract_features_batch_accepts_dict_result():
    original = rb.aiwaf_rust
    try:
        class Stub:
            def extract_features_batch_with_state(self, records, keywords, state):
                return {"features": [{"ip": "1.1.1.1"}], "state": {"cursor": 1}}

        rb.aiwaf_rust = Stub()
        feats, state = rb.extract_features_batch([{"ip": "1.1.1.1"}], [".env"], None)
        assert feats == [{"ip": "1.1.1.1"}]
        assert state == {"cursor": 1}
    finally:
        rb.aiwaf_rust = original


def test_extract_features_batch_accepts_tuple_result():
    original = rb.aiwaf_rust
    try:
        class Stub:
            def extract_features_batch_with_state(self, records, keywords, state):
                return ([{"ip": "2.2.2.2"}], {"cursor": 2})

        rb.aiwaf_rust = Stub()
        feats, state = rb.extract_features_batch([{"ip": "2.2.2.2"}], ["wp-"], None)
        assert feats == [{"ip": "2.2.2.2"}]
        assert state == {"cursor": 2}
    finally:
        rb.aiwaf_rust = original


def test_finalize_feature_state_extracts_features_from_dict():
    original = rb.aiwaf_rust
    try:
        class Stub:
            def finalize_feature_state(self, keywords, state):
                return {"features": [{"ip": "tail"}]}

        rb.aiwaf_rust = Stub()
        feats = rb.finalize_feature_state([".env"], {"cursor": 9})
        assert feats == [{"ip": "tail"}]
    finally:
        rb.aiwaf_rust = original
