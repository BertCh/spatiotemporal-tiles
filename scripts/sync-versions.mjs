#!/usr/bin/env node
/**
 * Version sync inside the cargo workspace.
 *
 * `[workspace.package] version` is one hand edit, but it is not the only place
 * the number appears: every internal path-dependency carries a `version = "…"`
 * alongside its `path = "../…"`, because cargo requires one for a published
 * dependency and it is the requirement consumers resolve. Left at the previous
 * release they do not merely go stale — `cargo update -w` fails outright
 * ("failed to select a version for the requirement `stt-core = ^0.5.0`"), which
 * is how the gap surfaced during the 0.6.0 cut.
 *
 * Canonical version = `[workspace.package] version` in the root `Cargo.toml`.
 * Before the 2026-08-26 repository split the canonical number lived in
 * `packages/core/package.json` and this file's headline job was keeping cargo
 * and npm level — they had diverged once (crates.io 0.4.0 against npm 0.5.0)
 * because the cargo bump was a hand edit no check covered. That failure mode is
 * gone rather than gated: the two stacks now release independently and are
 * related by the archive's `formatVersion`
 * (docs/roadmap/repo-split-2026-08.md §2.3). Do not re-add a cross-registry
 * comparison here.
 *
 * Usage:
 *   node scripts/sync-versions.mjs            # rewrite the stragglers in place
 *   node scripts/sync-versions.mjs --check    # report drift, exit 1 (CI gate)
 */
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

const ROOT = resolve(new URL('..', import.meta.url).pathname);
const CHECK = process.argv.slice(2).includes('--check');

const rel = (p) => relative(ROOT, p);
const read = (p) => readFileSync(p, 'utf8');

/** The number every other file must agree with. */
function canonicalVersion() {
  const table = cargoWorkspacePackageTable(read(join(ROOT, 'Cargo.toml')));
  const m = table && CARGO_VERSION_RE.exec(table.text);
  if (!m?.[1] && !m?.[2]) {
    throw new Error('Cargo.toml has no [workspace.package] version');
  }
  return m[2];
}

// ---------------------------------------------------------------------------
// Targets. `read()` returns one `{ label, value }` per version field the file
// owns; `write(text, want)` returns the new file text. Edits are surgical
// (anchored on the version key) rather than a parse → re-serialize round trip,
// so formatting, quote style, and key order survive.
// ---------------------------------------------------------------------------

/**
 * The `[workspace.package]` table of a cargo manifest, as `{ start, text }`
 * (null if absent). Scoping to the table matters: `[workspace.dependencies]`
 * below it is full of `version = "…"` lines for third-party crates, and a
 * file-wide match would rewrite arrow or serde to our version number.
 */
function cargoWorkspacePackageTable(text) {
  const start = text.search(/^\[workspace\.package\][ \t]*$/m);
  if (start === -1) return null;
  const body = text.slice(start);
  // Search past the table header's own `[` for the next table header.
  const next = body.slice(1).search(/^\[/m);
  return { start, text: next === -1 ? body : body.slice(0, next + 1) };
}

const CARGO_VERSION_RE = /^(version[ \t]*=[ \t]*")([^"]*)(")/m;

/** `version = "x.y.z"` under `[workspace.package]` in a cargo manifest. */
function cargoWorkspaceVersionTarget(file) {
  return {
    file,
    read() {
      const table = existsSync(file)
        ? cargoWorkspacePackageTable(read(file))
        : null;
      const m = table && CARGO_VERSION_RE.exec(table.text);
      return [
        { label: 'workspace.package.version', value: m ? m[2] : undefined },
      ];
    },
    write(text, want) {
      const table = cargoWorkspacePackageTable(text);
      if (!table) return text;
      const next = table.text.replace(
        CARGO_VERSION_RE,
        (_m, head, _old, tail) => `${head}${want}${tail}`,
      );
      return (
        text.slice(0, table.start) +
        next +
        text.slice(table.start + table.text.length)
      );
    },
  };
}

/**
 * The `version = "x.y.z"` inside every internal path-dependency of a member
 * manifest — `stt-core = { path = "../stt-core", version = "0.6.0" }`.
 *
 * These are separate from `workspace.package.version` and are NOT optional
 * decoration: cargo requires a version alongside `path` for a dependency that
 * is published, and it is the version REQUIREMENT consumers resolve. Left at
 * the previous release they do not merely go stale — `cargo update -w` fails
 * outright ("failed to select a version for the requirement `stt-core =
 * ^0.5.0`, candidate versions found which didn't match: 0.6.0"), which is how
 * this gap surfaced during the 0.6.0 cut. Matching on the `path = "../…"` shape
 * keeps this to first-party crates; third-party pins are never touched.
 */
const CARGO_PATH_DEP_RE =
  /^([ \t]*[\w-]+[ \t]*=[ \t]*\{[^}\n]*\bpath[ \t]*=[ \t]*"\.\.[^"]*"[^}\n]*\bversion[ \t]*=[ \t]*")([^"]*)(")/gm;

function cargoPathDepTarget(file) {
  const names = (text) =>
    [...text.matchAll(CARGO_PATH_DEP_RE)].map(
      (m) => m[1].trim().split(/[ \t]*=/)[0],
    );
  return {
    file,
    read() {
      if (!existsSync(file)) return [];
      const text = read(file);
      const found = [...text.matchAll(CARGO_PATH_DEP_RE)];
      return found.map((m, i) => ({
        label: `dependencies.${names(text)[i]}.version`,
        value: m[2],
      }));
    },
    write(text, want) {
      return text.replace(
        CARGO_PATH_DEP_RE,
        (_m, head, _old, tail) => `${head}${want}${tail}`,
      );
    },
  };
}

function collectTargets() {
  const targets = [cargoWorkspaceVersionTarget(join(ROOT, 'Cargo.toml'))];
  // Every workspace member's internal path-deps, in a stable order.
  const cratesDir = join(ROOT, 'crates');
  if (existsSync(cratesDir)) {
    for (const entry of readdirSync(cratesDir, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .sort((a, b) => a.name.localeCompare(b.name))) {
      const manifest = join(cratesDir, entry.name, 'Cargo.toml');
      if (existsSync(manifest)) targets.push(cargoPathDepTarget(manifest));
    }
  }
  return targets;
}

// ---------------------------------------------------------------------------

const want = canonicalVersion();
console.log(
  `canonical version: ${want}  ([workspace.package] in Cargo.toml)\n`,
);

let drift = 0;
let updated = 0;
let failed = 0;
for (const target of collectTargets()) {
  const stale = target.read().filter((f) => f.value !== want);
  const where = rel(target.file);

  if (stale.length === 0) {
    console.log(`  ok      ${where}`);
    continue;
  }
  drift += stale.length;
  const detail = stale
    .map((f) => `${f.label}=${f.value ?? '<missing>'}`)
    .join(', ');

  if (CHECK) {
    console.error(`  DRIFT   ${where} — ${detail} (want ${want})`);
    continue;
  }
  const text = read(target.file);
  const next = target.write(text, want);
  if (next === text) {
    failed += 1;
    console.error(
      `  FAILED  ${where} — ${detail}; no version field to rewrite`,
    );
    continue;
  }
  writeFileSync(target.file, next);
  updated += stale.length;
  console.log(`  wrote   ${where} — ${detail} → ${want}`);
}

if (CHECK) {
  if (drift > 0) {
    console.error(
      `\nsync-versions: ${drift} stale version field(s). Run \`node scripts/sync-versions.mjs\` and commit.`,
    );
    process.exit(1);
  }
  console.log('\nsync-versions: all in sync');
} else {
  console.log(
    `\nsync-versions: ${updated} field(s) updated, ${failed} unwritable`,
  );
  if (failed > 0) process.exit(1);
}
