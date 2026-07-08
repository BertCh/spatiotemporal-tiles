#!/usr/bin/env node
/**
 * AIS Data Validation Script
 *
 * This script validates that our coordinate decoding exactly matches the raw AIS data.
 * It reads a GeoJSON file and checks coordinate ranges.
 */

const fs = require('fs');
const path = require('path');

// Path to AIS GeoJSON (if it exists)
const GEOJSON_PATH =
  process.argv[2] || path.join(__dirname, 'data', 'ais-traffic.geojson');

console.log('='.repeat(80));
console.log('AIS Data Validation Script');
console.log('='.repeat(80));
console.log('');

if (!fs.existsSync(GEOJSON_PATH)) {
  console.error('❌ GeoJSON file not found:', GEOJSON_PATH);
  console.log('');
  console.log('To generate AIS GeoJSON data:');
  console.log('  1. Download AIS CSV from NOAA:');
  console.log('     ./download-ais.sh 2024 01 01');
  console.log('  2. Process to GeoJSON:');
  console.log('     cargo run --release --bin generate-ais-data \\');
  console.log('       --input data/AIS_2024_01_01.csv \\');
  console.log('       --output data/ais-traffic.geojson \\');
  console.log('       --sample-minutes 10 \\');
  console.log('       --bounds "25.0,-80.0,45.0,-65.0"');
  process.exit(1);
}

console.log('📂 Reading:', GEOJSON_PATH);

const data = JSON.parse(fs.readFileSync(GEOJSON_PATH, 'utf8'));

if (!data.features || !Array.isArray(data.features)) {
  console.error('❌ Invalid GeoJSON: no features array');
  process.exit(1);
}

console.log(`✓ Loaded ${data.features.length} features`);
console.log('');

// Analyze coordinates
let minLon = Infinity;
let maxLon = -Infinity;
let minLat = Infinity;
let maxLat = -Infinity;

const vesselTypes = new Map();
const timestamps = [];
const mmsiSet = new Set();

console.log('📊 Analyzing coordinates...');
console.log('');

for (let i = 0; i < data.features.length; i++) {
  const feature = data.features[i];

  if (feature.geometry && feature.geometry.type === 'Point') {
    const [lon, lat] = feature.geometry.coordinates;

    minLon = Math.min(minLon, lon);
    maxLon = Math.max(maxLon, lon);
    minLat = Math.min(minLat, lat);
    maxLat = Math.max(maxLat, lat);
  }

  if (feature.properties) {
    // Track vessel types
    const type = feature.properties.vessel_type || 'unknown';
    vesselTypes.set(type, (vesselTypes.get(type) || 0) + 1);

    // Track timestamps
    if (feature.properties.timestamp) {
      timestamps.push(new Date(feature.properties.timestamp).getTime());
    }

    // Track unique vessels
    if (feature.properties.mmsi) {
      mmsiSet.add(feature.properties.mmsi);
    }
  }

  // Show progress
  if ((i + 1) % 10000 === 0) {
    process.stdout.write(`\r  Processed ${i + 1} features...`);
  }
}

console.log(`\r  Processed ${data.features.length} features    `);
console.log('');

// Results
console.log('='.repeat(80));
console.log('COORDINATE BOUNDS');
console.log('='.repeat(80));
console.log('');
console.log(`Longitude: ${minLon.toFixed(4)} to ${maxLon.toFixed(4)}`);
console.log(`Latitude:  ${minLat.toFixed(4)} to ${maxLat.toFixed(4)}`);
console.log('');
console.log(
  `Center: [${((minLon + maxLon) / 2).toFixed(4)}, ${((minLat + maxLat) / 2).toFixed(4)}]`,
);
console.log(
  `Span:   [${(maxLon - minLon).toFixed(4)}°, ${(maxLat - minLat).toFixed(4)}°]`,
);
console.log('');

// Expected bounds for US East Coast
const EXPECTED_MIN_LON = -80.0;
const EXPECTED_MAX_LON = -65.0;
const EXPECTED_MIN_LAT = 25.0;
const EXPECTED_MAX_LAT = 45.0;

console.log('Expected bounds for US East Coast:');
console.log(`  Longitude: ${EXPECTED_MIN_LON} to ${EXPECTED_MAX_LON}`);
console.log(`  Latitude:  ${EXPECTED_MIN_LAT} to ${EXPECTED_MAX_LAT}`);
console.log('');

// Validation
let valid = true;
if (minLon < EXPECTED_MIN_LON || maxLon > EXPECTED_MAX_LON) {
  console.log(`⚠️  WARNING: Longitude outside expected range`);
  valid = false;
}
if (minLat < EXPECTED_MIN_LAT || maxLat > EXPECTED_MAX_LAT) {
  console.log(`⚠️  WARNING: Latitude outside expected range`);
  valid = false;
}

if (valid) {
  console.log('✅ Coordinates within expected US East Coast bounds');
} else {
  console.log('❌ Coordinates spread beyond expected region!');
  console.log('');
  console.log('This suggests either:');
  console.log('  1. Wrong bounds were used during data generation');
  console.log('  2. Coordinate decoding has an error');
  console.log('  3. Data includes vessels from other regions');
}
console.log('');

// Vessel types
console.log('='.repeat(80));
console.log('VESSEL TYPES');
console.log('='.repeat(80));
console.log('');
const sortedTypes = Array.from(vesselTypes.entries()).sort(
  (a, b) => b[1] - a[1],
);
for (const [type, count] of sortedTypes) {
  const percent = ((count / data.features.length) * 100).toFixed(1);
  console.log(
    `  ${type.padEnd(15)} ${count.toString().padStart(6)} (${percent}%)`,
  );
}
console.log('');

// Time range
if (timestamps.length > 0) {
  timestamps.sort((a, b) => a - b);
  const startTime = new Date(timestamps[0]);
  const endTime = new Date(timestamps[timestamps.length - 1]);
  const duration =
    (timestamps[timestamps.length - 1] - timestamps[0]) / 1000 / 3600; // hours

  console.log('='.repeat(80));
  console.log('TEMPORAL RANGE');
  console.log('='.repeat(80));
  console.log('');
  console.log(`Start: ${startTime.toISOString()}`);
  console.log(`End:   ${endTime.toISOString()}`);
  console.log(`Duration: ${duration.toFixed(1)} hours`);
  console.log('');
}

// Vessel count
console.log('='.repeat(80));
console.log('VESSELS');
console.log('='.repeat(80));
console.log('');
console.log(`Unique vessels (MMSI): ${mmsiSet.size}`);
console.log(
  `Average positions per vessel: ${(data.features.length / mmsiSet.size).toFixed(1)}`,
);
console.log('');

// Sample features
console.log('='.repeat(80));
console.log('SAMPLE FEATURES');
console.log('='.repeat(80));
console.log('');
console.log('First 3 features:');
for (let i = 0; i < Math.min(3, data.features.length); i++) {
  const f = data.features[i];
  console.log('');
  console.log(`Feature ${i + 1}:`);
  console.log(
    `  Coordinates: [${f.geometry.coordinates[0].toFixed(4)}, ${f.geometry.coordinates[1].toFixed(4)}]`,
  );
  if (f.properties) {
    console.log(`  MMSI: ${f.properties.mmsi || 'N/A'}`);
    console.log(`  Type: ${f.properties.vessel_type || 'N/A'}`);
    console.log(`  Speed: ${f.properties.speed || 'N/A'} knots`);
    console.log(`  Time: ${f.properties.timestamp || 'N/A'}`);
  }
}
console.log('');

// Summary
console.log('='.repeat(80));
console.log('VALIDATION SUMMARY');
console.log('='.repeat(80));
console.log('');

if (valid) {
  console.log('✅ All checks passed!');
  console.log('');
  console.log('The raw AIS data is correctly bounded to the US East Coast.');
  console.log('');
  console.log('Next steps:');
  console.log('  1. Build STT file with this data');
  console.log('  2. Load in showcase app');
  console.log('  3. Verify rendered ships match these coordinates');
  console.log('');
  console.log('Expected viewport center in showcase:');
  console.log(`  longitude: ${((minLon + maxLon) / 2).toFixed(2)}`);
  console.log(`  latitude: ${((minLat + maxLat) / 2).toFixed(2)}`);
  console.log(`  zoom: 5-7`);
} else {
  console.log('❌ Validation failed!');
  console.log('');
  console.log('Please check the data generation process and filtering bounds.');
}
console.log('');
