//! The SQL tokenizer must never panic on any text, including unusual Unicode.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = aiwaf_core::sql::classify(&text);
    let _ = aiwaf_core::sql::classify_fields(&text);
    let _ = aiwaf_core::sql::clean(&text);
});
