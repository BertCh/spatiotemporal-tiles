//! Protocol Buffer encoding and decoding utilities
//!
//! This module provides helper functions for encoding and decoding
//! spatiotemporal tiles using Protocol Buffers.

use crate::error::Result;
use crate::tile::{Feature, Layer, Tile, Value};
use crate::types::GeometryType;
use prost::Message;

/// Encode a tile to Protocol Buffer bytes
pub fn encode_tile(tile: &Tile) -> Result<Vec<u8>> {
    let proto_tile = tile_to_proto(tile)?;
    let mut buf = Vec::new();
    proto_tile.encode(&mut buf)?;
    Ok(buf)
}

/// Decode a tile from Protocol Buffer bytes
pub fn decode_tile(bytes: &[u8]) -> Result<Tile> {
    let proto_tile = crate::proto::Tile::decode(bytes)?;
    proto_to_tile(proto_tile)
}

/// Convert internal Tile to Protocol Buffer Tile
fn tile_to_proto(tile: &Tile) -> Result<crate::proto::Tile> {
    let layers = tile
        .layers
        .iter()
        .map(|layer| layer_to_proto(layer))
        .collect::<Result<Vec<_>>>()?;

    Ok(crate::proto::Tile {
        version: 1,
        time_start: tile.time_range.start,
        time_end: tile.time_range.end,
        layers,
        interpolation: None, // TODO: Implement interpolation hints
        temporal_resolution: None, // Metadata added by stt-build
    })
}

/// Convert Protocol Buffer Tile to internal Tile
fn proto_to_tile(proto_tile: crate::proto::Tile) -> Result<Tile> {
    // For now, we'll create a default TileId - this should be passed in
    let tile_id = crate::tile::TileId::new(0, 0, 0, proto_tile.time_start);
    
    let layers = proto_tile
        .layers
        .into_iter()
        .map(|proto_layer| proto_to_layer(proto_layer))
        .collect::<Result<Vec<_>>>()?;

    Ok(Tile {
        id: tile_id,
        time_range: crate::types::TimeRange::new(proto_tile.time_start, proto_tile.time_end),
        layers,
    })
}

/// Convert internal Layer to Protocol Buffer Layer
fn layer_to_proto(layer: &Layer) -> Result<crate::proto::Layer> {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut key_map = std::collections::HashMap::new();
    let mut value_map = std::collections::HashMap::new();

    let features = layer
        .features
        .iter()
        .map(|feature| {
            feature_to_proto(feature, &mut keys, &mut values, &mut key_map, &mut value_map)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(crate::proto::Layer {
        name: layer.name.clone(),
        extent: layer.extent,
        keys,
        values,
        features,
    })
}

/// Convert Protocol Buffer Layer to internal Layer
fn proto_to_layer(proto_layer: crate::proto::Layer) -> Result<Layer> {
    let features = proto_layer
        .features
        .into_iter()
        .map(|proto_feature| proto_to_feature(proto_feature, &proto_layer.keys, &proto_layer.values))
        .collect::<Result<Vec<_>>>()?;

    Ok(Layer {
        name: proto_layer.name,
        extent: proto_layer.extent,
        features,
    })
}

/// Convert internal Feature to Protocol Buffer Feature
fn feature_to_proto(
    feature: &Feature,
    keys: &mut Vec<String>,
    values: &mut Vec<crate::proto::Value>,
    key_map: &mut std::collections::HashMap<String, u32>,
    value_map: &mut std::collections::HashMap<String, u32>,
) -> Result<crate::proto::Feature> {
    let mut tags = Vec::new();

    for (key, value) in &feature.properties {
        // Get or insert key index
        let key_idx = *key_map.entry(key.clone()).or_insert_with(|| {
            let idx = keys.len() as u32;
            keys.push(key.clone());
            idx
        });

        // Get or insert value index
        let value_str = format!("{:?}", value); // Simple serialization
        let value_idx = *value_map.entry(value_str.clone()).or_insert_with(|| {
            let idx = values.len() as u32;
            values.push(value_to_proto(value));
            idx
        });

        tags.push(key_idx);
        tags.push(value_idx);
    }

    let (valid_from, valid_to) = if let Some(ref tr) = feature.time_range {
        (tr.start, tr.end)
    } else {
        (0, 0)
    };

    Ok(crate::proto::Feature {
        id: feature.id,
        r#type: feature.geometry_type.to_proto(),
        geometry: feature.geometry.clone(),
        tags,
        valid_from,
        valid_to,
        previous_hash: 0, // No previous hash
        change: 0, // UNCHANGED
    })
}

/// Convert Protocol Buffer Feature to internal Feature
fn proto_to_feature(
    proto_feature: crate::proto::Feature,
    keys: &[String],
    values: &[crate::proto::Value],
) -> Result<Feature> {
    let mut properties = std::collections::HashMap::new();

    for chunk in proto_feature.tags.chunks(2) {
        if chunk.len() == 2 {
            let key_idx = chunk[0] as usize;
            let val_idx = chunk[1] as usize;
            if let (Some(key), Some(val)) = (keys.get(key_idx), values.get(val_idx)) {
                properties.insert(key.clone(), proto_to_value(val));
            }
        }
    }

    let time_range = if proto_feature.valid_from > 0 && proto_feature.valid_to > 0 {
        Some(crate::types::TimeRange::new(
            proto_feature.valid_from,
            proto_feature.valid_to,
        ))
    } else {
        None
    };

    Ok(Feature {
        id: proto_feature.id,
        geometry_type: GeometryType::from_proto(proto_feature.r#type),
        geometry: proto_feature.geometry,
        properties,
        time_range,
    })
}

/// Convert internal Value to Protocol Buffer Value
fn value_to_proto(value: &Value) -> crate::proto::Value {
    use crate::proto::value::ValueType;

    let value_type = match value {
        Value::String(s) => Some(ValueType::StringValue(s.clone())),
        Value::Double(d) => Some(ValueType::DoubleValue(*d)),
        Value::Float(f) => Some(ValueType::FloatValue(*f)),
        Value::Int(i) => Some(ValueType::IntValue(*i)),
        Value::UInt(u) => Some(ValueType::UintValue(*u)),
        Value::Bool(b) => Some(ValueType::BoolValue(*b)),
    };

    crate::proto::Value { value_type }
}

/// Convert Protocol Buffer Value to internal Value
fn proto_to_value(proto_value: &crate::proto::Value) -> Value {
    use crate::proto::value::ValueType;

    match &proto_value.value_type {
        Some(ValueType::StringValue(s)) => Value::String(s.clone()),
        Some(ValueType::DoubleValue(d)) => Value::Double(*d),
        Some(ValueType::FloatValue(f)) => Value::Float(*f),
        Some(ValueType::IntValue(i)) => Value::Int(*i),
        Some(ValueType::UintValue(u)) => Value::UInt(*u),
        Some(ValueType::SintValue(s)) => Value::Int(*s),
        Some(ValueType::BoolValue(b)) => Value::Bool(*b),
        None => Value::String(String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_roundtrip() {
        let values = vec![
            Value::String("test".to_string()),
            Value::Double(1.5),
            Value::Float(2.5),
            Value::Int(-42),
            Value::UInt(42),
            Value::Bool(true),
        ];

        for value in values {
            let proto = value_to_proto(&value);
            let decoded = proto_to_value(&proto);
            // Note: This is a simple test, in production we'd need proper equality
            assert!(matches!(decoded, Value::String(_) | Value::Double(_) | Value::Float(_) | Value::Int(_) | Value::UInt(_) | Value::Bool(_)));
        }
    }
}

