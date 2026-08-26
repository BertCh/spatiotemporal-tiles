# Security

## Reporting a vulnerability

Report privately — do **not** open a public issue.

For usage questions, ordinary defects, and feature requests, follow
[SUPPORT.md](SUPPORT.md) instead. The security channel is only for suspected
vulnerabilities.

- Preferred: GitHub's private vulnerability reporting on this repository
  (**Security → Report a vulnerability**).
- Fallback, if that is unavailable: email <rgcgeog@gmail.com> with `[security]`
  in the subject.

This is a solo project. Expect an acknowledgement within about a week and a fix
on a best-effort schedule; there is no SLA. If a report needs coordinated
disclosure, say so and a date will be agreed before anything is published.

## Supported versions

Only the **latest published** version of each artifact gets fixes:

| Artifact                                                   | Registry  |
| ---------------------------------------------------------- | --------- |
| `spatiotemporal-tiles` (+ `stt-core`/`-build`/`-optimize`) | crates.io |

The project is pre-1.0, so "supported" means the current version — patches are
not backported to older 0.x lines. The `@poopdeck.gl/*` npm packages are a
separate project with its own policy; report renderer issues to
[BertCh/poopdeck.gl](https://github.com/BertCh/poopdeck.gl/security).

## What is in scope

The interesting surface is **untrusted archive bytes**. A `.stt` archive is
fetched over HTTP and decoded by code that runs in a user's browser or on their
machine, so the decoders treat every byte as hostile input:

- `stt-core` — manifest parsing, pack and directory decoding, zstd frames,
  quantized geometry. Memory-safety issues, panics that a remote archive can
  trigger, unbounded allocation from attacker-chosen lengths. (The TypeScript
  reader, `@poopdeck.gl/core`, is the same surface in another language and is
  reported downstream.)
- The CLIs `stt-build` / `stt-optimize` / `stt-validate` / `stt-bundle` — same
  decoders, plus input parsing (GeoParquet, WKB) of files you did not author.
- `stt-serve` — it binds a port. Anything that lets a request read outside the
  configured archive/table, or that turns a query parameter into unintended SQL
  against the Postgres or DuckDB backend.
- `stt-wasm` — the same decoders compiled for a third-party host.

## What is not in scope

- The demo datasets and the tile bucket that serves them: public read-only data,
  published deliberately.
- Anything in the renderer or the showcase site — including the URL-restricted
  map client tokens it ships. Report those to
  [BertCh/poopdeck.gl](https://github.com/BertCh/poopdeck.gl/security).
- Denial of service achieved by pointing the tools at a deliberately enormous
  local file. Building a 100 GB archive is a supported use, not an attack.
- Findings from automated scanners with no demonstrated impact on the above.
