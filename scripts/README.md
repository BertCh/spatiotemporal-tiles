# Scripts

Repository tooling for SpatioTemporal Tiles. Nothing here is published.

## Repository gates (Node)

| Script                        | What it does                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `check-project-status.mjs`    | Proves every claim in `project-status.json` against `Cargo.toml` and the version constants       |
| `sync-versions.mjs`           | The workspace version and every internal path-dependency's `version` agree                       |
| `check-doc-links.mjs`         | Every relative Markdown link resolves (cross-corpus rules in `docs/.corpus.json`)                |
| `gen-generate-datasets.mjs`   | Emits `docs/spec/stt-generate-datasets.json` from `stt-generate`'s clap enum; `--check` gates it |
| `patch-manifest-metadata.mjs` | Rewrites name/description/attribution in built manifests, in place                               |

## Publishing

| Script                | What it does                                                       |
| --------------------- | ------------------------------------------------------------------ |
| `r2-sync.sh`          | Uploads one archive stem to R2 with the two cache regimes          |
| `r2-cors.json`        | The bucket's CORS policy                                           |
| `rebuild-fleet-v3.sh` | Rebuilds the fleet beside the live one, with a feature-count guard |

The built fleet lives at `data-fleet/` (untracked, ~76 GB). It was
`examples/showcase/public/data` until the 2026-08-26 repository split moved the
showcase to [poopdeck.gl](https://github.com/BertCh/poopdeck.gl); a renderer
developer who wants it locally can symlink this directory into that checkout.

## Benchmarks

`postgis/` and `duckdb/` hold the ingest and serve benchmarks for the two
database input sources — see
[db-input-adaptors.md](../docs/roadmap/db-input-adaptors.md).

## Data Generation

Located in [`data-generation/`](./data-generation/), with helper scripts for generating showcase datasets.

### Quick Start

```bash
# Install the unified stt-generate tool
cargo install --path ../tools/stt-generate

# Generate all datasets
stt-generate all --output-dir ../data-fleet

# Or generate individually
stt-generate earthquakes --output earthquakes.stt
stt-generate hurricanes --output hurricanes.stt
stt-generate wildfires --output wildfires.stt
stt-generate ais --input ais.csv --output ais.stt
```

### Available Datasets

`stt-generate <subcommand>` covers earthquakes, AIS ship traffic, flights,
hurricanes, wildfires, NYC rideshare (+ taxi points/paths/trips/flows),
BIXI flowmaps, satellites, ocean drifters, animal migration, OSM edits, and
NEXRAD storm radar. Run `stt-generate --help` for the full registered list.

`emit_av_palettes.py` is not a generator: it exports the AV cockpit's palette
contract from `av_common.py` to `docs/spec/av-palettes.json`, which the renderer
vendors and asserts against. Run it with `--check` after touching a palette.

See [data-generation/README.md](./data-generation/README.md) and the
[Data Generation Guide](../docs/guides/data-generation.md) for per-dataset
recipes and flags.

---

**See also**: [Main Documentation](../README.md) | [Data Generation Guide](../docs/guides/data-generation.md)
