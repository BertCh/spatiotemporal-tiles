#!/usr/bin/env node
/**
 * Every relative link in every tracked Markdown file resolves.
 *
 * One exception, and it is declared rather than inferred. Since the 2026-08-26
 * repository split the published corpus is assembled from two repositories: the
 * pages listed in `docs/.corpus.json` as `vendoredDownstream` are copied into
 * poopdeck.gl and served from there alongside the renderer's own pages. A
 * vendored page may therefore link to a renderer page — `../api/stt-loader.md`
 * from the system overview — and that link resolves on the site and in the
 * downstream checkout, but not here.
 *
 * Rewriting those to absolute URLs would be worse: the site would link out of
 * itself for a page it is already serving. So the pairing is allowed, narrowly:
 * a VENDORED page may name a `downstreamOnly` path, and nothing else may. A
 * page this repository owns alone still has to keep every link local or make it
 * an absolute URL, and a vendored page linking to something that is in neither
 * list is still a failure.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const files = execFileSync(
  'git',
  ['ls-files', '--cached', '--others', '--exclude-standard', '*.md'],
  {
    cwd: ROOT,
    encoding: 'utf8',
  },
)
  .split('\n')
  .filter(Boolean)
  .sort();

const corpus = JSON.parse(
  readFileSync(resolve(ROOT, 'docs/.corpus.json'), 'utf8'),
);
const vendored = new Set(corpus.vendoredDownstream);
const downstreamOnly = corpus.downstreamOnly.map(
  (pattern) =>
    new RegExp(
      `^${pattern
        .split('*')
        .map((p) => p.replaceAll(/[.*+?^${}()|[\]\\]/gu, String.raw`\$&`))
        .join('[^/]*')}$`,
    ),
);

/** A path served only by the downstream repo's half of the corpus. */
function isDownstreamOnly(repoRelative) {
  return downstreamOnly.some((re) => re.test(repoRelative));
}

const failures = [];
let checked = 0;
let crossCorpus = 0;

for (const file of files) {
  const text = readFileSync(resolve(ROOT, file), 'utf8');
  for (const match of text.matchAll(/!?\[[^\]]*\]\(([^)]+)\)/g)) {
    let target = match[1].trim();
    if (target.startsWith('<') && target.endsWith('>')) {
      target = target.slice(1, -1);
    } else {
      target = target.split(/\s+["']/u, 1)[0];
    }
    if (
      !target ||
      target.startsWith('#') ||
      target.startsWith('/') ||
      /^[a-z][a-z0-9+.-]*:/iu.test(target)
    ) {
      continue;
    }

    const pathOnly = target.split('#', 1)[0].split('?', 1)[0];
    if (!pathOnly) continue;
    checked += 1;
    let decoded;
    try {
      decoded = decodeURIComponent(pathOnly);
    } catch {
      failures.push(`${file}: invalid URL encoding in ${target}`);
      continue;
    }
    if (existsSync(resolve(ROOT, dirname(file), decoded))) continue;

    const asRepoPath = relative(ROOT, resolve(ROOT, dirname(file), decoded));
    if (vendored.has(file) && isDownstreamOnly(asRepoPath)) {
      // Resolves wherever the corpus is whole, which is the published site.
      crossCorpus += 1;
      continue;
    }
    failures.push(
      isDownstreamOnly(asRepoPath)
        ? `${file}: ${target} lives only in ${corpus.downstreamRepo} — this file is not vendored, so use an absolute URL`
        : `${file}: missing ${target}`,
    );
  }
}

if (failures.length > 0) {
  console.error(`documentation links: ${failures.length} missing target(s)`);
  for (const failure of failures) console.error(`  ${failure}`);
  process.exit(1);
}

console.log(
  `documentation links: ${checked} relative targets resolve across ${files.length} Markdown files` +
    (crossCorpus > 0
      ? ` (${crossCorpus} into ${corpus.downstreamRepo}'s half of the corpus, from vendored pages)`
      : ''),
);
