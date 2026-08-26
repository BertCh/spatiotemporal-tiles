#!/usr/bin/env node
/**
 * Emit `docs/spec/stt-generate-datasets.json` — the subcommand inventory of the
 * repository-only `stt-generate`, derived from its clap `enum Commands`.
 *
 * Why an artifact. `@poopdeck.gl/mcp`'s `generate_dataset` tool advertises a zod
 * enum of dataset names that must equal this list exactly; a generator added or
 * removed here otherwise leaves an MCP client either unable to reach a real
 * generator, or offered one that makes `stt-generate` exit with "unrecognized
 * subcommand". That was gated by a TypeScript test that read `main.rs` directly
 * — which the 2026-08-26 repository split ended, since the two files no longer
 * share a tree (docs/roadmap/repo-split-2026-08.md §4).
 *
 * The parser moved HERE rather than the source moving there: a scanner for Rust
 * belongs beside the Rust it scans, and a stale artifact now fails in the
 * repository that caused it instead of in someone else's CI.
 *
 * Usage:
 *   node scripts/gen-generate-datasets.mjs           # rewrite the artifact
 *   node scripts/gen-generate-datasets.mjs --check   # byte-compare (CI gate)
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

const ROOT = resolve(new URL('..', import.meta.url).pathname);
const MAIN_RS = join(ROOT, 'tools/stt-generate/src/main.rs');
const ARTIFACT = join(ROOT, 'docs/spec/stt-generate-datasets.json');
const CHECK = process.argv.slice(2).includes('--check');

/**
 * heck's `to_kebab_case`, which is what clap's derive uses to turn a
 * `Subcommand` variant name into its CLI subcommand name.
 * `NycTaxiPoints` -> `nyc-taxi-points`, `DriftersHourly` -> `drifters-hourly`,
 * `Ais` -> `ais`, `OsmEdits` -> `osm-edits`.
 */
function variantToSubcommand(variant) {
  return variant
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1-$2')
    .toLowerCase();
}

/**
 * Extracts the top-level variant names of `enum Commands { … }`.
 *
 * Scans characters rather than matching line shapes: variants come in both
 * tuple (`Earthquakes(datasets::earthquakes::Args)`) and struct
 * (`All { output_dir: PathBuf, … }`) form, and the `///` doc comments contain
 * parentheses and quotes. Comments and string literals are skipped, and only
 * text at brace-depth 1 is kept, so nested field lists and `#[arg(…)]`
 * attributes fall out.
 */
function parseSubcommandVariants(src) {
  const enumIdx = src.indexOf('enum Commands {');
  if (enumIdx < 0)
    throw new Error('could not find `enum Commands {` in main.rs');
  let i = src.indexOf('{', enumIdx) + 1;

  let depth = 1;
  let out = '';
  let inString = false;
  let inLineComment = false;
  for (; i < src.length && depth > 0; i++) {
    const ch = src[i];
    if (inLineComment) {
      if (ch === '\n') inLineComment = false;
      continue;
    }
    if (inString) {
      if (ch === '\\') i++;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === '/' && src[i + 1] === '/') {
      inLineComment = true;
      i++;
      continue;
    }
    if (ch === '{' || ch === '(' || ch === '[') {
      depth++;
      continue;
    }
    if (ch === '}' || ch === ')' || ch === ']') {
      depth--;
      if (depth === 0) break;
      continue;
    }
    if (depth === 1) out += ch;
  }
  if (depth !== 0) throw new Error('unbalanced braces parsing `enum Commands`');

  const variants = [];
  for (const raw of out.split(',')) {
    // Attribute bodies were dropped as nested `[…]`; a bare `#` may remain.
    const segment = raw.replace(/#/g, '').trim();
    if (segment === '') continue;
    // A segment we cannot read is a PARSER failure, not an empty result — fail
    // loudly rather than silently under-reporting the variant set.
    if (!/^[A-Z][A-Za-z0-9_]*$/.test(segment)) {
      throw new Error(
        `unparsed segment in enum Commands: ${JSON.stringify(segment)}`,
      );
    }
    variants.push(segment);
  }
  return variants;
}

if (!existsSync(MAIN_RS)) {
  console.error(`gen-generate-datasets: ${relative(ROOT, MAIN_RS)} not found`);
  process.exit(1);
}

const src = readFileSync(MAIN_RS, 'utf8');

// The kebab-case derivation above is only valid while clap is actually doing
// it. A `#[command(name = "…")]` or `#[command(rename_all = "…")]` on a variant
// takes the naming away from the derive, at which point this file would emit a
// confidently WRONG list and every consumer would agree with it. Fail instead.
{
  const start = src.indexOf('#[derive(Subcommand)]');
  if (start < 0) throw new Error('expected #[derive(Subcommand)] in main.rs');
  const block = src.slice(
    start,
    src.indexOf('\n}', src.indexOf('enum Commands {')),
  );
  if (
    /#\[\s*(?:command|clap)\s*\([^\]]*\b(?:name|rename_all)\s*=/.test(block)
  ) {
    throw new Error(
      'a clap name/rename_all override was added to `enum Commands` — ' +
        'variantToSubcommand() no longer derives the real subcommand names ' +
        'and must be updated to match',
    );
  }
}

const datasets = parseSubcommandVariants(src).map(variantToSubcommand).sort();

const want =
  JSON.stringify(
    {
      $comment:
        'GENERATED by scripts/gen-generate-datasets.mjs from tools/stt-generate/src/main.rs — do not hand-edit. The subcommand inventory of the repository-only reference-dataset generator. @poopdeck.gl/mcp advertises exactly this list as `generate_dataset`’s dataset enum.',
      version: 1,
      command: 'stt-generate',
      datasets,
    },
    null,
    2,
  ) + '\n';

if (CHECK) {
  if (!existsSync(ARTIFACT)) {
    console.error(
      `gen-generate-datasets: ${relative(ROOT, ARTIFACT)} does not exist — run without --check`,
    );
    process.exit(1);
  }
  if (readFileSync(ARTIFACT, 'utf8') !== want) {
    console.error(
      `gen-generate-datasets: ${relative(ROOT, ARTIFACT)} is stale.\n` +
        '  `enum Commands` changed without regenerating. Run:\n' +
        '    node scripts/gen-generate-datasets.mjs',
    );
    process.exit(1);
  }
  console.log(
    `stt-generate-datasets.json matches main.rs (${datasets.length} datasets)`,
  );
} else {
  writeFileSync(ARTIFACT, want);
  console.log(
    `wrote ${relative(ROOT, ARTIFACT)} (${datasets.length} datasets)`,
  );
}
