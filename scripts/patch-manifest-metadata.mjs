#!/usr/bin/env node
/**
 * Patch the human-facing metadata of a packed archive's `manifest.json` in
 * place — `name`, `description`, `attribution` — without rebuilding it.
 *
 * ## Why this is safe
 *
 * The packed format content-addresses `index/**` and `packs/**` by blake3, and
 * those bytes must never be rewritten. `manifest.json` is the one MUTABLE
 * object in an archive (that is why `r2-sync.sh` gives it a 60 s TTL while the
 * rest ship `immutable`), and these three fields are free text that nothing
 * indexes, hashes, or decodes against. Editing them changes no address, so a
 * re-sync uploads exactly one tiny object and every open session keeps working.
 *
 * ## Why it exists
 *
 * `tiles.poopdeck.gl/data/earthquakes-v2/manifest.json` is the first file
 * anyone following the quickstart fetches, and it said:
 *
 *     "name": "earthquakes-v2.new", "description": "", "attribution": ""
 *
 * — the scratch name from whatever build produced it, and no credit for the
 * USGS data it carries. Any UI that surfaces `metadata.name`, including this
 * project's own, displayed the build artifact (DX review 2026-08-26, F9).
 *
 * ## Usage
 *
 *   # Report the whole local fleet's metadata state and exit.
 *   node scripts/patch-manifest-metadata.mjs --scan
 *
 *   # Patch one dataset (any subset of the three fields).
 *   node scripts/patch-manifest-metadata.mjs earthquakes-v2 \
 *     --name "Global earthquakes M4.0+ (2020-2024)" \
 *     --description "..." \
 *     --attribution "USGS Earthquake Catalog (ComCat), public domain"
 *
 * `--dir` overrides the dataset root (default
 * `examples/showcase/public/data`). Publishing the change is a separate,
 * deliberate step: `scripts/r2-sync.sh <stem>`.
 */

import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const FIELDS = ['name', 'description', 'attribution'];

const argv = process.argv.slice(2);
const flags = new Map();
const positional = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i].startsWith('--')) {
    const key = argv[i].slice(2);
    if (key === 'scan') flags.set('scan', true);
    else flags.set(key, argv[++i]);
  } else {
    positional.push(argv[i]);
  }
}

const dataDir = resolve(
  ROOT,
  flags.get('dir') ?? 'examples/showcase/public/data',
);

function readManifest(stem) {
  const path = join(dataDir, stem, 'manifest.json');
  if (!existsSync(path)) return null;
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(path, 'utf8'));
  } catch {
    return null;
  }
  if (manifest?.format !== 'stt-packed') return null;
  return { path, manifest };
}

if (flags.get('scan')) {
  const stems = readdirSync(dataDir, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort();
  let scratch = 0;
  let noDescription = 0;
  let noAttribution = 0;
  const rows = [];
  for (const stem of stems) {
    const found = readManifest(stem);
    if (!found) continue;
    const md = found.manifest.metadata ?? {};
    const name = md.name ?? '';
    // A name that is empty, carries a build suffix, or simply is not the stem
    // is the build's scratch label rather than something written for a reader.
    const looksScratch = name === '' || name !== stem;
    if (looksScratch) scratch++;
    if (!md.description) noDescription++;
    if (!md.attribution) noAttribution++;
    rows.push({
      stem,
      name,
      looksScratch,
      description: Boolean(md.description),
      attribution: Boolean(md.attribution),
    });
  }
  for (const r of rows) {
    const marks = [
      r.looksScratch ? 'name' : '    ',
      r.description ? '    ' : 'desc',
      r.attribution ? '    ' : 'attr',
    ].join(' ');
    console.log(`${marks}  ${r.stem.padEnd(32)} ${JSON.stringify(r.name)}`);
  }
  console.log(
    `\n${rows.length} packed datasets under ${dataDir}\n` +
      `  ${scratch} carry a build-scratch name\n` +
      `  ${noDescription} have no description\n` +
      `  ${noAttribution} have no attribution`,
  );
  process.exit(0);
}

const stem = positional[0];
if (!stem) {
  console.error(
    'usage: patch-manifest-metadata.mjs <stem> [--name S] [--description S] [--attribution S]\n' +
      '       patch-manifest-metadata.mjs --scan',
  );
  process.exit(2);
}

const found = readManifest(stem);
if (!found) {
  console.error(
    `no packed manifest at ${join(dataDir, stem, 'manifest.json')}`,
  );
  process.exit(1);
}

const { path, manifest } = found;
manifest.metadata ??= {};
let changed = 0;
for (const field of FIELDS) {
  const next = flags.get(field);
  if (next === undefined) continue;
  const prev = manifest.metadata[field] ?? '';
  if (prev === next) {
    console.log(`  ${field}: unchanged`);
    continue;
  }
  manifest.metadata[field] = next;
  console.log(`  ${field}: ${JSON.stringify(prev)} → ${JSON.stringify(next)}`);
  changed++;
}

if (changed === 0) {
  console.log(`${stem}: nothing to change`);
  process.exit(0);
}

// The writer emits two-space-indented JSON and — as of the shipped fleet — NO
// trailing newline. Re-serialize with the same indent and preserve whichever
// trailing convention the file already had, so the diff is the patched fields
// and nothing else.
const original = readFileSync(path, 'utf8');
writeFileSync(
  path,
  JSON.stringify(manifest, null, 2) + (original.endsWith('\n') ? '\n' : ''),
);
console.log(
  `\n${stem}: patched ${changed} field(s) in ${path}\n` +
    `Publish with: scripts/r2-sync.sh ${stem}   (manifest.json is the only object that changes)`,
);
