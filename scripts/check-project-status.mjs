#!/usr/bin/env node
/**
 * `project-status.json` is this repository's published claim about what it
 * ships and what the reference writer emits. This gate proves every claim
 * against the file that actually decides it — the cargo manifest, the two
 * version constants in `stt-core`, the facade's bin table — so the document
 * can never quietly describe a previous release.
 *
 * Scope is HALF the pre-split check. It also used to verify the pnpm version,
 * the Node floors and the eight `@poopdeck.gl/*` package versions; those
 * sources left with the renderer on 2026-08-26. The poopdeck.gl repository
 * runs the identical gate over its half and vendors THIS document as its
 * `stt` block, byte-compared rather than re-derived
 * (docs/roadmap/repo-split-2026-08.md §4.4).
 */

import { readFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const status = json('project-status.json');

function read(relativePath) {
  return readFileSync(join(ROOT, relativePath), 'utf8');
}

function json(relativePath) {
  return JSON.parse(read(relativePath));
}

function capture(text, pattern, label) {
  const value = pattern.exec(text)?.[1];
  if (value === undefined) throw new Error(`could not read ${label}`);
  return value;
}

function assertEqual(actual, expected, label) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(
      `${label} drifted: project-status=${JSON.stringify(actual)}, source=${JSON.stringify(expected)}`,
    );
  }
}

const cargo = read('Cargo.toml');
const rustVersion = capture(
  cargo,
  /^version = "([^"]+)"$/m,
  'Cargo workspace version',
);
const rustMsrv = capture(cargo, /^rust-version = "([^"]+)"$/m, 'Rust MSRV');
assertEqual(status.release.rust, rustVersion, 'Rust release');
assertEqual(status.toolchain.rust, rustMsrv, 'Rust MSRV');

const packSource = read('crates/stt-core/src/pack/mod.rs');
const directorySource = read('crates/stt-core/src/directory.rs');
const writeFormat = Number(
  capture(
    packSource,
    /pub const PACKED_FORMAT_VERSION: u32 = ([0-9]+);/,
    'packed format version',
  ),
);
const minFormat = Number(
  capture(
    packSource,
    /pub const MIN_PACKED_FORMAT_VERSION: u32 = ([0-9]+);/,
    'minimum packed format version',
  ),
);
const writeDirectory = Number(
  capture(
    directorySource,
    /pub const DIRECTORY_VERSION: u8 = ([0-9]+);/,
    'directory version',
  ),
);
const minDirectory = Number(
  capture(
    directorySource,
    /pub const MIN_DIRECTORY_VERSION: u8 = ([0-9]+);/,
    'minimum directory version',
  ),
);
assertEqual(status.archive.writes.formatVersion, writeFormat, 'writer format');
assertEqual(
  status.archive.writes.directoryVersion,
  writeDirectory,
  'writer directory',
);
assertEqual(
  status.archive.reads.formatVersions,
  range(minFormat, writeFormat),
  'reader format window',
);
assertEqual(
  status.archive.reads.directoryVersions,
  range(minDirectory, writeDirectory),
  'reader directory window',
);

function range(start, end) {
  return Array.from({ length: end - start + 1 }, (_, index) => start + index);
}

const facadeCargo = read('crates/spatiotemporal-tiles/Cargo.toml');
const facadeCommands = [
  ...facadeCargo.matchAll(/^name = "(stt-[^"]+)"$/gm),
].map((match) => match[1]);
assertEqual(
  status.commands.map(({ name }) => name).sort(),
  facadeCommands.sort(),
  'published command inventory',
);
assertEqual(
  status.repositoryOnlyCommands.map(({ name }) => name),
  ['stt-generate'],
  'repository-only command inventory',
);

const schemaPath = status.$schema;
if (basename(schemaPath) !== 'project-status.schema.json') {
  throw new Error(`unexpected project-status schema: ${schemaPath}`);
}
json(schemaPath.replace(/^\.\//, ''));

console.log(
  `project status: ${facadeCommands.length} CLIs, format ${writeFormat}/directory ${writeDirectory}, rust ${rustVersion} (MSRV ${rustMsrv}); all in sync`,
);
