use aiwaf_wasm::{
    IsolationForest, analyze_recent_behavior, build_records, extract_features,
    extract_features_batch_with_state, finalize_feature_state, python_feature_from_record,
    python_features_batched, rust_payload_from_records, validate_headers,
};
use js_sys::Array;
use serde_wasm_bindgen::{from_value, to_value};
use std::collections::HashMap;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;
use web_sys::Headers;

#[wasm_bindgen_test]
fn test_validate_headers() {
    let headers = serde_wasm_bindgen::to_value(&serde_json::json!({
        "HTTP_USER_AGENT": "Mozilla/5.0",
        "HTTP_ACCEPT": "text/html",
        "HTTP_ACCEPT_LANGUAGE": "en-US",
        "HTTP_ACCEPT_ENCODING": "gzip, deflate",
        "HTTP_CONNECTION": "keep-alive"
    }))
    .unwrap();
    let reason = validate_headers(headers).unwrap();
    let opt: Option<String> = from_value(reason).unwrap();
    assert!(opt.is_none());
}

#[wasm_bindgen_test]
fn test_validate_headers_with_headers_object() {
    if web_sys::window().is_none() {
        // Headers iteration is not reliably supported in wasm-bindgen-test --node.
        return;
    }
    let headers = Headers::new().unwrap();
    headers.set("user-agent", "Mozilla/5.0").unwrap();
    headers.set("accept", "text/html").unwrap();
    headers.set("accept-language", "en-US").unwrap();
    headers.set("accept-encoding", "gzip, deflate").unwrap();
    headers.set("connection", "keep-alive").unwrap();

    let reason = validate_headers(JsValue::from(headers)).unwrap();
    let raw: JsValue = reason.clone();
    let opt: Option<String> = from_value(reason).unwrap();
    if opt.is_some() {
        // Surface the exact reason to help diagnose header conversion issues.
        let s = raw
            .as_string()
            .unwrap_or_else(|| "<non-string>".to_string());
        panic!("unexpected reason: {s}");
    }
}

#[wasm_bindgen_test]
fn test_extract_features_and_state() {
    let records = to_value(&vec![aiwaf_core::FeatureRecordInput {
        ip: "1.2.3.4".to_string(),
        path_lower: "/wp-admin".to_string(),
        path_len: 9,
        timestamp: 10.0,
        response_time: 0.03,
        status_idx: 3,
        kw_check: true,
        total_404: 5,
    }])
    .unwrap();
    let keywords = serde_wasm_bindgen::to_value(&vec!["wp"]).unwrap();
    let out = extract_features(records, keywords).unwrap();
    assert!(out.is_object());

    let records = to_value(&vec![aiwaf_core::FeatureRecordInput {
        ip: "1.2.3.4".to_string(),
        path_lower: "/wp-admin".to_string(),
        path_len: 9,
        timestamp: 10.0,
        response_time: 0.03,
        status_idx: 3,
        kw_check: true,
        total_404: 5,
    }])
    .unwrap();
    let keywords = serde_wasm_bindgen::to_value(&vec!["wp"]).unwrap();
    let batch = extract_features_batch_with_state(records, keywords, JsValue::NULL).unwrap();
    assert!(batch.is_object());
    let _state = finalize_feature_state().unwrap();
}

#[wasm_bindgen_test]
fn test_training_record_helpers() {
    let parsed = to_value(&vec![
        aiwaf_core::ParsedFeatureRecord {
            ip: "1.2.3.4".to_string(),
            path: "/wp-admin".to_string(),
            response_time: 0.03,
            status: 404,
            timestamp: 10.0,
        },
        aiwaf_core::ParsedFeatureRecord {
            ip: "1.2.3.4".to_string(),
            path: "/wp-admin".to_string(),
            response_time: 0.04,
            status: 500,
            timestamp: 12.0,
        },
    ])
    .unwrap();
    let ip_404 = to_value(&HashMap::from([("1.2.3.4".to_string(), 2)])).unwrap();
    let exists = js_sys::Function::new_with_args("path", "return false;");
    let exempt = js_sys::Function::new_with_args("path", "return false;");
    let statuses = to_value(&vec![200, 404]).unwrap();

    let records = build_records(
        parsed,
        ip_404,
        JsValue::from(exists),
        JsValue::from(exempt),
        statuses,
    )
    .unwrap();
    let built: Vec<aiwaf_core::BuiltFeatureRecord> = from_value(records.clone()).unwrap();
    assert_eq!(built.len(), 2);
    assert_eq!(built[0].path_lower, "/wp-admin");
    assert_eq!(built[0].path_len, 9);
    assert_eq!(built[0].status_idx, 1);
    assert_eq!(built[1].status_idx, -1);
    assert!(built[0].kw_check);
    assert_eq!(built[0].total_404, 2);

    let payload = rust_payload_from_records(records.clone()).unwrap();
    let payload: Vec<aiwaf_core::FeatureRecordInput> = from_value(payload).unwrap();
    assert_eq!(payload[0].timestamp, 10.0);
    assert_eq!(payload[0].response_time, 0.03);

    let feature = python_feature_from_record(
        Array::from(&records).get(0),
        to_value(&HashMap::from([("1.2.3.4".to_string(), vec![1.0, 20.0])])).unwrap(),
        to_value(&vec!["wp"]).unwrap(),
    )
    .unwrap();
    let feature: aiwaf_core::FeatureRecordOutput = from_value(feature).unwrap();
    assert_eq!(feature.kw_hits, 1);
    assert_eq!(feature.burst_count, 2);

    let features = python_features_batched(
        records,
        to_value(&HashMap::from([("1.2.3.4".to_string(), vec![10.0])])).unwrap(),
        to_value(&vec!["wp"]).unwrap(),
        JsValue::NULL,
        1,
        false,
        1,
        1,
    )
    .unwrap();
    let features: Vec<aiwaf_core::FeatureRecordOutput> = from_value(features).unwrap();
    assert_eq!(features.len(), 2);
}

#[wasm_bindgen_test]
fn test_analyze_recent_behavior() {
    let entries = to_value(&vec![
        aiwaf_core::RecentEntryInput {
            path_lower: "/wp-admin".to_string(),
            timestamp: 1.0,
            status: 404,
            kw_check: true,
        },
        aiwaf_core::RecentEntryInput {
            path_lower: "/.env".to_string(),
            timestamp: 2.0,
            status: 404,
            kw_check: true,
        },
    ])
    .unwrap();
    let keywords = serde_wasm_bindgen::to_value(&vec!["wp"]).unwrap();
    let res = analyze_recent_behavior(entries, keywords).unwrap();
    assert!(res.is_object() || res.is_null());
}

#[wasm_bindgen_test]
fn test_isolation_forest_roundtrip() {
    let config = serde_wasm_bindgen::to_value(&serde_json::json!({
        "n_estimators": 10,
        "max_samples": 8,
        "contamination": "auto",
        "max_features": 1.0,
        "bootstrap": false,
        "random_state": 7,
        "warm_start": false
    }))
    .unwrap();
    let mut forest = IsolationForest::new(Some(config)).unwrap();
    let data = to_value(&vec![vec![0.0], vec![1.0], vec![2.0], vec![10.0]]).unwrap();
    forest.fit(data).unwrap();
    let score = forest
        .anomaly_score(to_value(&vec![10.0]).unwrap())
        .unwrap();
    assert!(score >= 0.0);
    let state = forest.to_json().unwrap();
    let mut forest2 = IsolationForest::from_json(state).unwrap();
    forest2
        .retrain(to_value(&vec![vec![0.1], vec![0.2], vec![9.5]]).unwrap())
        .unwrap();
}
