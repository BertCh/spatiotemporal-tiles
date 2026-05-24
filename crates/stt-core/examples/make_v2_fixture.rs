//! Build a legacy `STT\x02` archive equivalent to `make_sample_fixture`,
//! for v2-vs-v3 size and decode comparisons via `tools/bench --compare`.

use std::path::PathBuf;

use stt_core::{
    archive::ArchiveWriter,
    arrow_tile::{encode_tile, ColumnarLayer, GeometryColumn, PropertyColumn},
    metadata::Metadata,
    tile::TileId,
    types::{BoundingBox, Compression, TimeRange},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: make_v2_fixture <output.stt>")?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let kinds = ["car", "bus", "scooter", "bike"];
    let feature_count = 25usize;
    let layer = ColumnarLayer {
        name: "default".to_string(),
        feature_ids: (0..feature_count as u64).collect(),
        start_times: (0..feature_count)
            .map(|i| 1_700_000_000_000 + (i as i64) * 1000)
            .collect(),
        end_times: (0..feature_count)
            .map(|i| 1_700_000_000_000 + (i as i64) * 1000 + 500)
            .collect(),
        geometry: GeometryColumn::Point(
            (0..feature_count)
                .map(|i| {
                    [
                        -122.42 + ((i as f64) % 5.0) * 0.005,
                        37.77 + ((i / 5) as f64) * 0.005,
                    ]
                })
                .collect(),
        ),
        vertex_times: None,
        triangles: None,
        properties: vec![
            (
                "speed".into(),
                PropertyColumn::Numeric(
                    (0..feature_count)
                        .map(|i| Some(10.0 + (i as f64)))
                        .collect(),
                ),
            ),
            (
                "kind".into(),
                PropertyColumn::Categorical(
                    (0..feature_count)
                        .map(|i| Some(kinds[i % kinds.len()].to_string()))
                        .collect(),
                ),
            ),
        ],
    };

    let payload = encode_tile(&[layer])?;

    let mut writer = ArchiveWriter::create_v2(&out, Compression::Gzip)?;
    writer.add_tile(
        &TileId::new(10, 163, 395, 1_700_000_000_000),
        1_700_000_000_000,
        1_700_000_100_000,
        feature_count as u32,
        &payload,
    )?;

    let metadata = Metadata::new("sample-fixture-v2")
        .with_bounds(BoundingBox::new(-122.5, 37.7, -122.3, 37.9))
        .with_time_range(TimeRange::new(1_700_000_000_000, 1_700_000_100_000))
        .with_zoom_levels(10, 12)
        .with_temporal_bucket_ms(3_600_000);
    writer.finalize(&metadata)?;

    println!("wrote v2 fixture {}", out.display());
    Ok(())
}
