//! Geometry quantization for efficient coordinate storage
//!
//! This module provides variable-length integer encoding and
//! advanced quantization strategies to minimize coordinate storage.

/// Variable-length integer encoding (protobuf-style varint)
pub mod varint {
    /// Encode a u32 as a varint
    pub fn encode_u32(mut value: u32) -> Vec<u8> {
        let mut result = Vec::new();
        
        while value >= 0x80 {
            result.push((value as u8 & 0x7F) | 0x80);
            value >>= 7;
        }
        result.push(value as u8);
        
        result
    }

    /// Decode a varint to u32
    pub fn decode_u32(bytes: &[u8]) -> (u32, usize) {
        let mut result = 0u32;
        let mut shift = 0;
        let mut pos = 0;

        for &byte in bytes {
            pos += 1;
            result |= ((byte & 0x7F) as u32) << shift;
            
            if byte & 0x80 == 0 {
                break;
            }
            
            shift += 7;
            if shift >= 32 {
                break;
            }
        }

        (result, pos)
    }

    /// Encode a signed integer with zigzag encoding
    pub fn encode_i32_zigzag(value: i32) -> Vec<u8> {
        let encoded = ((value << 1) ^ (value >> 31)) as u32;
        encode_u32(encoded)
    }

    /// Decode a zigzag-encoded signed integer
    pub fn decode_i32_zigzag(bytes: &[u8]) -> (i32, usize) {
        let (encoded, size) = decode_u32(bytes);
        let decoded = ((encoded >> 1) as i32) ^ (-((encoded & 1) as i32));
        (decoded, size)
    }
}

/// Quantization strategy for coordinates
#[derive(Debug, Clone, Copy)]
pub enum QuantizationLevel {
    /// No quantization - use full precision
    None,
    /// Low precision (16 bits per coordinate)
    Low,
    /// Medium precision (24 bits per coordinate)
    Medium,
    /// High precision (32 bits per coordinate, default)
    High,
}

impl QuantizationLevel {
    /// Get the number of bits for this quantization level
    pub fn bits(&self) -> u8 {
        match self {
            Self::None => 32,
            Self::Low => 16,
            Self::Medium => 24,
            Self::High => 32,
        }
    }

    /// Get the scale factor for this level
    pub fn scale(&self) -> u32 {
        match self {
            Self::None => 1,
            Self::Low => 1 << 16,
            Self::Medium => 1 << 24,
            Self::High => 1 << 30,
        }
    }
}

/// Quantized geometry encoder
pub struct GeometryQuantizer {
    /// Quantization level
    level: QuantizationLevel,
    /// Coordinate extent (typically 4096 for MVT compatibility)
    extent: u32,
}

impl GeometryQuantizer {
    /// Create a new quantizer
    pub fn new(level: QuantizationLevel, extent: u32) -> Self {
        Self { level, extent }
    }

    /// Encode coordinates with delta encoding and varint compression
    pub fn encode_coordinates(&self, coords: &[u32]) -> Vec<u8> {
        if coords.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut last_x = 0i32;
        let mut last_y = 0i32;

        // Process coordinate pairs
        for chunk in coords.chunks(2) {
            if chunk.len() == 2 {
                let x = chunk[0] as i32;
                let y = chunk[1] as i32;

                // Calculate deltas
                let dx = x - last_x;
                let dy = y - last_y;

                // Encode deltas as varints with zigzag encoding
                result.extend(varint::encode_i32_zigzag(dx));
                result.extend(varint::encode_i32_zigzag(dy));

                last_x = x;
                last_y = y;
            }
        }

        result
    }

    /// Decode coordinates from varint-encoded deltas
    pub fn decode_coordinates(&self, encoded: &[u8]) -> Vec<u32> {
        let mut coords = Vec::new();
        let mut x = 0i32;
        let mut y = 0i32;
        let mut pos = 0;

        while pos < encoded.len() {
            // Decode X delta
            let (dx, dx_size) = varint::decode_i32_zigzag(&encoded[pos..]);
            pos += dx_size;
            
            if pos >= encoded.len() {
                break;
            }

            // Decode Y delta
            let (dy, dy_size) = varint::decode_i32_zigzag(&encoded[pos..]);
            pos += dy_size;

            x += dx;
            y += dy;

            coords.push(x as u32);
            coords.push(y as u32);
        }

        coords
    }

    /// Calculate the estimated compressed size for coordinates
    pub fn estimate_size(&self, coords: &[u32]) -> usize {
        if coords.is_empty() {
            return 0;
        }

        let mut size = 0;
        let mut last_x = 0i32;
        let mut last_y = 0i32;

        for chunk in coords.chunks(2) {
            if chunk.len() == 2 {
                let x = chunk[0] as i32;
                let y = chunk[1] as i32;

                let dx = x - last_x;
                let dy = y - last_y;

                size += varint::encode_i32_zigzag(dx).len();
                size += varint::encode_i32_zigzag(dy).len();

                last_x = x;
                last_y = y;
            }
        }

        size
    }
}

/// Adaptive quantization based on coordinate distribution
pub struct AdaptiveQuantizer {
    /// Minimum coordinate value
    min_x: f64,
    min_y: f64,
    /// Maximum coordinate value
    max_x: f64,
    max_y: f64,
    /// Scale factor
    scale: f64,
}

impl AdaptiveQuantizer {
    /// Create a new adaptive quantizer from coordinate bounds
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64, precision: u32) -> Self {
        let range_x = max_x - min_x;
        let range_y = max_y - min_y;
        let max_range = range_x.max(range_y);
        
        let scale = if max_range > 0.0 {
            precision as f64 / max_range
        } else {
            1.0
        };

        Self {
            min_x,
            min_y,
            max_x,
            max_y,
            scale,
        }
    }

    /// Quantize a coordinate pair
    pub fn quantize(&self, x: f64, y: f64) -> (u32, u32) {
        let qx = ((x - self.min_x) * self.scale) as u32;
        let qy = ((y - self.min_y) * self.scale) as u32;
        (qx, qy)
    }

    /// Dequantize a coordinate pair
    pub fn dequantize(&self, qx: u32, qy: u32) -> (f64, f64) {
        let x = (qx as f64 / self.scale) + self.min_x;
        let y = (qy as f64 / self.scale) + self.min_y;
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encoding() {
        // Test small values (1 byte)
        assert_eq!(varint::encode_u32(0), vec![0]);
        assert_eq!(varint::encode_u32(127), vec![127]);

        // Test medium values (2 bytes)
        assert_eq!(varint::encode_u32(128), vec![0x80, 0x01]);
        assert_eq!(varint::encode_u32(300), vec![0xAC, 0x02]);

        // Test larger values
        assert_eq!(varint::encode_u32(16384), vec![0x80, 0x80, 0x01]);
    }

    #[test]
    fn test_varint_decoding() {
        assert_eq!(varint::decode_u32(&[0]), (0, 1));
        assert_eq!(varint::decode_u32(&[127]), (127, 1));
        assert_eq!(varint::decode_u32(&[0x80, 0x01]), (128, 2));
        assert_eq!(varint::decode_u32(&[0xAC, 0x02]), (300, 2));
    }

    #[test]
    fn test_varint_roundtrip() {
        for value in [0, 1, 127, 128, 255, 256, 1000, 10000, 100000, u32::MAX].iter() {
            let encoded = varint::encode_u32(*value);
            let (decoded, _) = varint::decode_u32(&encoded);
            assert_eq!(decoded, *value);
        }
    }

    #[test]
    fn test_zigzag_encoding() {
        assert_eq!(varint::encode_i32_zigzag(0), vec![0]);
        assert_eq!(varint::encode_i32_zigzag(-1), vec![1]);
        assert_eq!(varint::encode_i32_zigzag(1), vec![2]);
        assert_eq!(varint::encode_i32_zigzag(-2), vec![3]);
    }

    #[test]
    fn test_zigzag_roundtrip() {
        for value in [-1000, -100, -10, -1, 0, 1, 10, 100, 1000].iter() {
            let encoded = varint::encode_i32_zigzag(*value);
            let (decoded, _) = varint::decode_i32_zigzag(&encoded);
            assert_eq!(decoded, *value);
        }
    }

    #[test]
    fn test_coordinate_quantization() {
        let quantizer = GeometryQuantizer::new(QuantizationLevel::High, 4096);
        
        let coords = vec![100, 200, 150, 250, 200, 300];
        let encoded = quantizer.encode_coordinates(&coords);
        let decoded = quantizer.decode_coordinates(&encoded);
        
        assert_eq!(decoded, coords);
        
        // Verify compression (encoded should be smaller than raw u32 storage)
        assert!(encoded.len() < coords.len() * 4);
    }

    #[test]
    fn test_adaptive_quantizer() {
        let quantizer = AdaptiveQuantizer::new(0.0, 0.0, 100.0, 100.0, 4096);
        
        let (qx, qy) = quantizer.quantize(50.0, 75.0);
        let (x, y) = quantizer.dequantize(qx, qy);
        
        // Should be approximately equal (within quantization error)
        assert!((x - 50.0).abs() < 0.1);
        assert!((y - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_delta_compression_efficiency() {
        let quantizer = GeometryQuantizer::new(QuantizationLevel::High, 4096);
        
        // Coordinates with small deltas (typical for lines/polygons)
        let coords = vec![
            1000, 2000,
            1001, 2001,
            1002, 2002,
            1003, 2003,
            1004, 2004,
        ];
        
        let encoded = quantizer.encode_coordinates(&coords);
        
        // With small deltas, varint encoding should be very efficient
        // Each delta of 1 should encode to 1 byte (zigzag(1) = 2, varint(2) = [0x02])
        assert!(encoded.len() < coords.len() * 2); // Much better than u32 storage
    }
}

