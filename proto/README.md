# Protocol Buffer Definitions

This directory contains the Protocol Buffer schema definitions for the SpatioTemporal Tiles format.

## Files

- **tile.proto**: Core tile format with features, geometries, and temporal metadata
- **index.proto**: Spatial and temporal index structures for efficient queries
- **metadata.proto**: Archive metadata including bounds, statistics, and generation info

## Generating Code

### Rust

```bash
# Install protoc if not already installed
# macOS: brew install protobuf
# Linux: apt-get install protobuf-compiler

# Generate Rust code (handled by build.rs in crates)
cargo build
```

### TypeScript

```bash
# From the repo root, run:
pnpm run generate-proto

# Or manually:
npx pbjs -t static-module -w es6 --es6 -o packages/core/src/proto.js proto/*.proto
npx pbts -o packages/core/src/proto.d.ts packages/core/src/proto.js
```

**Note**: Run this whenever you change the .proto files to regenerate the TypeScript bindings.

## Design Decisions

### Compatibility with Mapbox Vector Tiles

The tile format is intentionally similar to Mapbox Vector Tiles (MVT) for several reasons:

1. **Proven Format**: MVT is battle-tested at scale
2. **Efficient Encoding**: Delta-encoded geometries are very compact
3. **Tool Ecosystem**: Many tools can already parse MVT-like structures
4. **Easy Migration**: Existing MVT tiles can be converted to STT

### Key Differences from MVT

1. **Temporal Metadata**: Each tile has time_start/time_end
2. **Change Tracking**: Features include change_type for delta encoding
3. **Interpolation Hints**: Metadata for smooth temporal transitions
4. **Feature Validity**: Optional valid_from/valid_to per feature

### Spatial Indexing: Hilbert Curve

We use a Hilbert curve for spatial indexing because:

1. **Spatial Locality**: Nearby tiles have nearby indices
2. **Cache Friendly**: Better cache performance than Z-order curves
3. **Range Queries**: Efficient bounding box queries
4. **Deterministic**: Same tiles always get same indices

### Temporal Indexing: B-tree Style

The temporal index uses a sorted timestamp array with offsets because:

1. **Binary Search**: O(log n) time range queries
2. **Compact**: Much smaller than a full B-tree structure
3. **Predictable**: Good for prefetching during animation

### Compression

Tiles can be compressed with:

1. **Gzip** (recommended): Good compression ratio, fast decode, universal support
2. **None**: For pre-compressed data or debugging

The index and metadata are stored uncompressed for fast access.

## Wire Format

The STT archive uses a simple container format:

```
┌─────────────────────────────────────┐
│ Magic Number (7 bytes)              │  "STT\x01\x00\x00\x00"
├─────────────────────────────────────┤
│ Version (1 byte)                    │  Format version (currently 1)
├─────────────────────────────────────┤
│ Index Offset (8 bytes, uint64)      │  Byte offset to index
├─────────────────────────────────────┤
│ Index Length (8 bytes, uint64)      │  Index size in bytes
├─────────────────────────────────────┤
│ Metadata Offset (8 bytes, uint64)   │  Byte offset to metadata
├─────────────────────────────────────┤
│ Metadata Length (8 bytes, uint64)   │  Metadata size in bytes
├─────────────────────────────────────┤
│ Reserved (16 bytes)                 │  For future use
├─────────────────────────────────────┤
│                                     │
│ Tile Data (variable)                │  Compressed tiles
│                                     │
├─────────────────────────────────────┤
│ Index (Protocol Buffer)             │  Spatial/temporal index
├─────────────────────────────────────┤
│ Metadata (JSON or Protocol Buffer)  │  Archive metadata
└─────────────────────────────────────┘
```

Total header size: 53 bytes

### Why This Format?

1. **HTTP Range Requests**: Fixed header allows fetching metadata first
2. **Appendable**: Can build archive incrementally
3. **Self-Describing**: Metadata is at a known offset
4. **Simple**: No complex container format needed

## Examples

### Creating a Tile

```rust
use stt::proto::{Tile, Layer, Feature, GeomType};

let tile = Tile {
    version: 1,
    time_start: 1609459200000, // 2021-01-01
    time_end: 1609545600000,   // 2021-01-02
    layers: vec![
        Layer {
            name: "points".to_string(),
            extent: 4096,
            features: vec![
                Feature {
                    id: 1,
                    type: GeomType::Point as i32,
                    geometry: vec![9, 2048, 2048], // MoveTo(1024, 1024)
                    tags: vec![0, 0], // key[0] = value[0]
                    ..Default::default()
                }
            ],
            keys: vec!["name".to_string()],
            values: vec![/* ... */],
        }
    ],
    ..Default::default()
};
```

### Reading a Tile (TypeScript)

```typescript
import { Tile } from './generated/proto';

const buffer = await fetch(url, {
  headers: { Range: `bytes=${offset}-${offset + length - 1}` }
}).then(r => r.arrayBuffer());

const tile = Tile.decode(new Uint8Array(buffer));

console.log(`Tile covers: ${new Date(tile.time_start)} to ${new Date(tile.time_end)}`);
```

## Versioning

The Protocol Buffer schemas use field numbers that allow backward-compatible evolution:

- **Never change** field numbers
- **Always add** new fields with new numbers
- **Use reserved** for deprecated fields
- **Keep required** fields to minimum (prefer optional)

## Performance Notes

- Protocol Buffer decoding: ~1-2ms for typical tile (5-10KB compressed)
- Index size: ~0.1% of total archive size
- Metadata size: ~1KB regardless of archive size

---

**Version**: 1.0
**Last Updated**: October 24, 2025

