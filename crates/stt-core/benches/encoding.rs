//! Benchmarks for the hot path: Arrow tile encode / decode.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stt_core::arrow_tile::{
    decode_tile, encode_tile, ColumnarLayer, GeometryColumn, PropertyColumn,
};

/// A point layer with `n` features and one numeric + one categorical column.
fn point_layer(n: usize) -> ColumnarLayer {
    let kinds = ["car", "bus", "bike", "tram"];
    ColumnarLayer {
        polygon_parts: None,
        name: "default".to_string(),
        feature_ids: (0..n as u64).collect(),
        start_times: (0..n as i64)
            .map(|i| 1_600_000_000_000 + i * 1000)
            .collect(),
        end_times: (0..n as i64)
            .map(|i| 1_600_000_000_000 + i * 1000 + 500)
            .collect(),
        geometry: GeometryColumn::Point(
            (0..n)
                .map(|i| [-122.4 + (i as f64) * 1e-4, 37.7 + (i as f64) * 1e-4])
                .collect(),
        ),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            (
                "speed".to_string(),
                PropertyColumn::Numeric((0..n).map(|i| Some(i as f64 % 60.0)).collect()),
            ),
            (
                "kind".to_string(),
                PropertyColumn::Categorical(
                    (0..n).map(|i| Some(kinds[i % 4].to_string())).collect(),
                ),
            ),
        ],
    }
}

/// A linestring layer with `n` features of `verts` vertices each.
fn line_layer(n: usize, verts: usize) -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: "tracks".to_string(),
        feature_ids: (0..n as u64).collect(),
        start_times: vec![1_600_000_000_000; n],
        end_times: vec![1_600_000_003_600; n],
        geometry: GeometryColumn::LineString(
            (0..n)
                .map(|f| {
                    (0..verts)
                        .map(|v| [(f + v) as f64 * 1e-3, (f + v) as f64 * 1e-3])
                        .collect()
                })
                .collect(),
        ),
        vertex_times: Some(
            (0..n)
                .map(|_| (0..verts as i64).map(|v| v * 100).collect())
                .collect(),
        ),
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    }
}

fn bench_encoding(c: &mut Criterion) {
    let points = point_layer(1000);
    let encoded_points = encode_tile(&[points.clone()]).unwrap();
    c.bench_function("encode_tile/1000_points", |b| {
        b.iter(|| encode_tile(black_box(std::slice::from_ref(&points))).unwrap())
    });
    c.bench_function("decode_tile/1000_points", |b| {
        b.iter(|| decode_tile(black_box(&encoded_points)).unwrap())
    });

    let lines = line_layer(200, 50);
    let encoded_lines = encode_tile(&[lines.clone()]).unwrap();
    c.bench_function("encode_tile/200_lines_x50v", |b| {
        b.iter(|| encode_tile(black_box(std::slice::from_ref(&lines))).unwrap())
    });
    c.bench_function("decode_tile/200_lines_x50v", |b| {
        b.iter(|| decode_tile(black_box(&encoded_lines)).unwrap())
    });
}

criterion_group!(benches, bench_encoding);
criterion_main!(benches);
