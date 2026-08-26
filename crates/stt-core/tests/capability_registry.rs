//! Cross-language pin for the `manifest.capabilities` registry.
//!
//! The registry exists in two reference implementations (Rust
//! `KNOWN_CAPABILITIES`, TS `KNOWN_MANIFEST_CAPABILITIES`) plus the spec.
//! Its single machine-readable source of truth is the top-level
//! `x-stt-capability-registry` array in `docs/spec/manifest.schema.json`;
//! this test pins the Rust constant against it and the TS suite
//! (`poopdeck:packages/core/test/manifest-schema.test.ts`) pins the TS constant
//! against the same array — so a registry addition on either side fails CI
//! until the schema (and therefore both sides) agree.

use stt_core::pack::KNOWN_CAPABILITIES;

#[test]
fn known_capabilities_match_schema_registry() {
    // The schema lives in the monorepo, not in the published crate tarball —
    // skip (loudly) when it is absent so `cargo test` still passes on a
    // crates.io checkout.
    let schema_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/spec/manifest.schema.json"
    );
    let Ok(schema_json) = std::fs::read_to_string(schema_path) else {
        eprintln!("capability_registry: schema not present (published-crate build) — skipping");
        return;
    };
    let schema: serde_json::Value =
        serde_json::from_str(&schema_json).expect("manifest.schema.json parses");
    let registry: Vec<&str> = schema["x-stt-capability-registry"]
        .as_array()
        .expect("schema carries the x-stt-capability-registry array")
        .iter()
        .map(|v| v.as_str().expect("registry entries are strings"))
        .collect();

    let mut known: Vec<&str> = KNOWN_CAPABILITIES.to_vec();
    known.sort_unstable();
    let mut from_schema = registry.clone();
    from_schema.sort_unstable();
    assert_eq!(
        known, from_schema,
        "KNOWN_CAPABILITIES and the schema's x-stt-capability-registry drifted — \
         update docs/spec/manifest.schema.json (and the TS reader's registry) together"
    );
}
