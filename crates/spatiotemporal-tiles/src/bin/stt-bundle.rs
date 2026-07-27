//! stt-bundle — single-file `.sttb` interchange bundles for packed datasets.
//!
//! `pack` folds an exploded packed dataset (`manifest.json` + its
//! content-addressed objects) into ONE file for hand-off — the "download one
//! file" property the packed layout gave up (PMTiles-style interchange);
//! `unpack` explodes it back. Objects round-trip **byte-identical**: they
//! are content-addressed, `pack` re-hashes each one on the way in, and
//! `unpack` re-verifies the result with the same integrity pass
//! `stt-validate` runs (any mismatch exits non-zero).
//!
//! Strictly an interchange profile (spec §13): the CDN/serving story is the
//! exploded layout — nothing serves bundles over HTTP ranges. The container is
//! object-agnostic, so it carries any manifest `formatVersion` unchanged.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "stt-bundle",
    version,
    about = "Pack a packed STT dataset into a single-file .sttb bundle, or unpack one",
    after_help = "\
The bundle is an INTERCHANGE profile: serve the exploded layout (manifest + \
content-addressed objects) in production — nothing serves bundles over HTTP \
ranges. Packing is deterministic (the same dataset packs to byte-identical \
bundle bytes), and `unpack` re-verifies every object's blake3 content \
address, so a pack→unpack round-trip is provably byte-identical. \
`stt-validate` accepts a .sttb directly."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pack an exploded packed dataset into one .sttb file
    Pack {
        /// Packed dataset directory or its manifest.json
        input: PathBuf,

        /// Output bundle path (conventionally <dataset>.sttb)
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Unpack a .sttb bundle back into an exploded packed dataset directory
    Unpack {
        /// Bundle file (.sttb)
        input: PathBuf,

        /// Output dataset directory (created if missing)
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Pack { input, output } => pack(&input, &output),
        Commands::Unpack { input, output } => unpack(&input, &output),
    }
}

/// If `input` denotes a packed dataset, return its `manifest.json` path
/// (mirrors stt-validate's input handling).
fn packed_manifest_path(input: &Path) -> Option<PathBuf> {
    if input.is_dir() {
        let manifest = input.join("manifest.json");
        return manifest.is_file().then_some(manifest);
    }
    if input
        .file_name()
        .map(|f| f == "manifest.json")
        .unwrap_or(false)
    {
        return Some(input.to_path_buf());
    }
    None
}

fn pack(input: &Path, output: &Path) -> Result<()> {
    let Some(manifest_path) = packed_manifest_path(input) else {
        anyhow::bail!(
            "{} is not a packed STT dataset (expected a directory containing \
             manifest.json, or the manifest.json itself)",
            input.display()
        );
    };
    let summary = stt_core::pack::write_bundle(&manifest_path, output).with_context(|| {
        format!(
            "packing {} into {}",
            manifest_path.display(),
            output.display()
        )
    })?;
    println!(
        "packed {} object(s) into {} ({} bytes)",
        summary.objects,
        output.display(),
        summary.bytes
    );
    Ok(())
}

fn unpack(input: &Path, output: &Path) -> Result<()> {
    let summary = stt_core::pack::unpack_bundle(input, output)
        .with_context(|| format!("unpacking {} into {}", input.display(), output.display()))?;
    // Round-trip proof: every unpacked object must re-verify its blake3
    // content address (the same integrity pass stt-validate runs). An
    // unpack that cannot verify exits non-zero and says why.
    let issues = stt_core::verify_packed_objects(output.join("manifest.json"))
        .context("verifying the unpacked dataset")?;
    if !issues.is_empty() {
        for issue in &issues {
            eprintln!("error: integrity: {issue}");
        }
        anyhow::bail!(
            "unpacked dataset failed integrity verification ({} issue(s))",
            issues.len()
        );
    }
    println!(
        "unpacked {} object(s) + manifest.json into {} (content addresses verified)",
        summary.objects,
        output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Doc gate: every visible long flag on every subcommand must appear in the
    /// `stt-bundle` section of `docs/api/cli-reference.md`, so a new flag fails
    /// the build until it is documented.
    #[test]
    fn cli_flags_are_documented_in_cli_reference() {
        use clap::CommandFactory;
        let doc = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/api/cli-reference.md"
        ))
        .expect("read docs/api/cli-reference.md");
        let start = doc
            .find("## `stt-bundle`")
            .expect("stt-bundle section heading");
        let body = &doc[start + 1..];
        let end = body
            .find("\n## `")
            .map(|i| start + 1 + i)
            .unwrap_or(doc.len());
        let section = &doc[start..end];
        let cmd = Cli::command();
        let mut missing: Vec<String> = Vec::new();
        for sub in cmd.get_subcommands() {
            for a in sub.get_arguments() {
                if a.is_hide_set() {
                    continue;
                }
                let Some(l) = a.get_long() else { continue };
                if matches!(l, "help" | "version") {
                    continue;
                }
                let flag = format!("--{l}");
                if !section.contains(flag.as_str()) {
                    missing.push(format!("{} {flag}", sub.get_name()));
                }
            }
        }
        missing.sort();
        missing.dedup();
        assert!(
            missing.is_empty(),
            "flags missing from the `stt-bundle` section of docs/api/cli-reference.md \
             (document them before shipping): {missing:?}"
        );
    }

    /// pack → unpack through the CLI-facing functions round-trips a real
    /// dataset and the traversal/malformed-input errors surface as errors,
    /// not panics.
    #[test]
    fn pack_and_unpack_roundtrip_via_cli_helpers() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("dataset");
        build_fixture(&src);

        let bundle = dir.path().join("dataset.sttb");
        pack(&src, &bundle).unwrap();
        // Also accepts the manifest.json path directly.
        pack(&src.join("manifest.json"), &dir.path().join("again.sttb")).unwrap();
        assert_eq!(
            std::fs::read(&bundle).unwrap(),
            std::fs::read(dir.path().join("again.sttb")).unwrap(),
            "bundling must be deterministic"
        );

        let out = dir.path().join("unpacked");
        unpack(&bundle, &out).unwrap();
        assert!(stt_core::verify_packed_objects(out.join("manifest.json"))
            .unwrap()
            .is_empty());

        // Not a dataset → loud error.
        assert!(pack(&dir.path().join("nope"), &dir.path().join("x.sttb")).is_err());
        // Not a bundle → loud error.
        assert!(unpack(&src.join("manifest.json"), &dir.path().join("y")).is_err());
    }

    /// Minimal packed dataset for the round-trip test: the
    /// payloads are bare `encode_tile` frames and the bundle container is
    /// format-version-agnostic (`add_tile_full` enforces frame/manifest
    /// coherence, so the writer must not inherit the v2 default).
    fn build_fixture(out: &Path) {
        use stt_core::arrow_tile::{encode_tile, ColumnarLayer, GeometryColumn};
        use stt_core::metadata::Metadata;
        use stt_core::{BlobOrdering, PackWriter, TileId};
        let mut w = PackWriter::create(out, BlobOrdering::Auto, 8 * 1024)
            .unwrap()
            .with_format_version(stt_core::pack::PACKED_FORMAT_VERSION);
        for k in 0..6u64 {
            let ids: Vec<u64> = (0..4).map(|i| k * 10 + i).collect();
            let n = ids.len();
            let payload = encode_tile(&[ColumnarLayer {
                polygon_parts: None,
                name: "default".to_string(),
                feature_ids: ids,
                start_times: vec![0; n],
                end_times: vec![100; n],
                geometry: GeometryColumn::Point(vec![[-73.5 + k as f64, 45.5]; n]),
                vertex_times: None,
                vertex_values: None,
                triangles: None,
                vertex_value_matrix: None,
                properties: vec![],
            }])
            .unwrap();
            w.add_tile_full(
                &TileId::new(10, k as u32, 0, 0),
                0,
                100,
                None,
                n as u32,
                None,
                &payload,
            )
            .unwrap();
        }
        w.finalize(&Metadata::new("bundle-cli-fixture")).unwrap();
    }
}
