//! Doc gate: `docs/api/cli-reference.md` and the shipped CLIs must agree, in
//! BOTH directions.
//!
//! The reference doc is hand-written prose + tables, so nothing but a test can
//! stop it from drifting the moment someone adds a clap arg. This suite runs
//! each compiled binary — and every subcommand it advertises, to any depth —
//! with `--help`, and asserts:
//!
//! 1. **binary → doc** (`stt_*_flags_are_documented`): every long flag clap
//!    exposes appears somewhere in the doc. A plain (word-bounded) substring
//!    search over the whole file is deliberate: a flag documented in a table
//!    cell, in a prose sentence, or under another binary's section all count as
//!    documented — the gate exists to catch *undocumented* flags, not to police
//!    where they are written up.
//! 2. **doc → binary** (`documented_flags_all_exist`): every long flag the doc
//!    writes inside `code` — either as a bare mention (`` `--publish` ``) or on
//!    a command line whose first word is one of our own CLIs — is a flag some
//!    binary really has. Without this a renamed or deleted flag rots in the doc
//!    indefinitely, since direction 1 can only ever notice *additions*.
//!
//! Direction 2 doubles as the backstop for direction 1's help-output parser: if
//! the scraper stops matching clap's layout for a binary, that binary's
//! real-flag set collapses and everything the doc documents for it surfaces as
//! a phantom — a loud failure, where direction 1 on its own would silently
//! check fewer flags (its only guard is a per-binary "parsed at least one").
//!
//! Adding a flag? Document it in `docs/api/cli-reference.md` and this passes.
//! Renaming or removing one? Update its doc mention in the same commit.
//!
//! Complements (does not replace) the per-binary `#[cfg(test)]` gates inside
//! `src/bin/*`, which introspect clap in-process and demand each flag land in
//! *its own* section of the doc. Those cannot see subcommand trees they do not
//! own, cannot see `stt-generate` at all, and match by bare substring; this
//! suite drives the real compiled binaries — so it also pins the surface as
//! actually feature-compiled — walks their subcommands, checks whole words,
//! covers `stt-generate`, and checks the reverse direction.
//!
//! Coverage note: the binaries live in this package, so Cargo hands us their
//! paths via `CARGO_BIN_EXE_<name>`. `stt-generate` does not — it lives at
//! `tools/stt-generate` in its OWN workspace (kept out of this one so its dep
//! tree and MSRV stay off the published crates), so no `CARGO_BIN_EXE_` exists
//! here AND no root-workspace build produces it.
//! `stt_generate_subcommands_are_documented` resolves it as a sibling of the
//! test binary in the target profile dir and reports a loud skip when it is
//! not there — which is now the normal case. To actually exercise the
//! generator half of this gate the binary has to be put in the root target dir
//! first, e.g.
//!
//! ```text
//! cargo build --manifest-path tools/stt-generate/Cargo.toml
//! cp tools/stt-generate/target/debug/stt-generate target/debug/
//! ```
//!
//! ⚠ A STALE `target/<profile>/stt-generate` left behind by an older build
//! satisfies that lookup and will silently check the doc against obsolete
//! `--help` output. Delete it rather than trusting a green run.
//!
//! Either way it is the subcommand level only, because the doc deliberately
//! defers the per-dataset flags to `stt-generate <sub> --help` (see that test).

// Every check below is `#[cfg]`-gated on the feature its binary needs (see the
// `required-features` on each `[[bin]]`); compile the file out entirely when
// none of them are on so a `--no-default-features` build stays clean.
#![cfg(any(
    feature = "build-cli",
    feature = "optimize-cli",
    feature = "validate-cli",
    feature = "bundle-cli",
    feature = "serve-core",
))]

use std::path::{Path, PathBuf};
use std::process::Command;

/// Clap synthesises these; they are not part of the documented surface.
const BUILTIN_FLAGS: &[&str] = &["--help", "--version"];

/// Our own CLI names. In the doc→binary direction a code span only contributes
/// flags when it *is* a flag or when its first word is one of these, so the
/// `--release` in an install snippet's `cargo build --release …` is never
/// mistaken for a phantom `stt-*` flag. (Structural, not an allow-list: a new
/// `pnpm …` or `curl …` example needs no maintenance here.)
const CLI_NAMES: &[&str] = &[
    "stt-build",
    "stt-optimize",
    "stt-validate",
    "stt-bundle",
    "stt-serve",
    "stt-generate",
];

/// Depth cap for the subcommand walk. Nothing nests today (`stt-optimize
/// analyze`, `stt-bundle pack`, `stt-generate storms` are all one level), so
/// the walk is fully general and this is only a guard against a pathological
/// or self-referential clap tree wedging the test.
const MAX_SUBCOMMAND_DEPTH: usize = 4;

/// The reference doc, resolved from the manifest dir (never the CWD — Cargo
/// does not promise which directory a test binary runs in).
fn reference_doc() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("docs")
        .join("api")
        .join("cli-reference.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read the CLI reference at {}: {e}", path.display()))
}

/// Run `<exe> <subcommand…> --help` and return its output.
fn help_text(exe: &str, subcommand: &[&str]) -> String {
    let out = Command::new(exe)
        .args(subcommand)
        .arg("--help")
        .output()
        .unwrap_or_else(|e| panic!("cannot run `{exe} {} --help`: {e}", subcommand.join(" ")));
    // Clap prints help on stdout and exits 0; take stderr too so a future
    // change of stream does not silently empty the gate.
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    assert!(
        text.contains("Usage:"),
        "`{exe} {} --help` produced no clap help output",
        subcommand.join(" ")
    );
    text
}

/// Split clap help into its `Heading:` sections. Headings sit at column 0 and a
/// section runs until the next column-0 non-empty line, which also drops the
/// free-form after-help prose (it may mention flags belonging to other tools).
fn section<'a>(help: &'a str, heading: &str) -> Vec<&'a str> {
    let mut lines = Vec::new();
    let mut inside = false;
    for line in help.lines() {
        let at_col0 = !line.is_empty() && !line.starts_with(' ');
        if at_col0 {
            inside = line.trim_end() == heading;
            continue;
        }
        if inside {
            lines.push(line);
        }
    }
    lines
}

/// Every `Heading:` in the help, so `help_heading`-grouped args are covered too.
fn headings(help: &str) -> Vec<&str> {
    help.lines()
        .filter(|l| !l.is_empty() && !l.starts_with(' '))
        .map(str::trim_end)
        .filter(|l| l.ends_with(':') && l.len() > 1)
        .filter(|l| *l != "Usage:" && *l != "Commands:")
        .collect()
}

/// True when `line` is an option-list entry rather than a wrapped description.
///
/// Clap indents option specs by 2 (`  -i, --input …`) or 6 (`      --publish …`,
/// padded past the short-flag column) and wraps long help at 10+, so bullet
/// continuations such as `          - unix-ms: …` are excluded twice over: by
/// the indent, and by requiring the char after the leading `-` to start a flag
/// name rather than a space.
fn is_option_line(line: &str) -> bool {
    let indent = line.len() - line.trim_start().len();
    if indent > 6 {
        return false;
    }
    let mut chars = line.trim_start().chars();
    chars.next() == Some('-')
        && matches!(chars.next(), Some(c) if c == '-' || c.is_ascii_alphanumeric())
}

/// Pull the long flags (`--like-this`) out of arbitrary text, skipping clap's
/// synthesised builtins and de-duplicating.
fn scan_flags(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut flags = Vec::new();
    let mut i = 0;
    while i + 2 < bytes.len() {
        // A run of hyphens (markdown table rule, `---`) is never a flag: the
        // char after `--` must open a flag name.
        let opens = bytes[i] == b'-'
            && bytes[i + 1] == b'-'
            && (bytes[i + 2].is_ascii_lowercase() || bytes[i + 2].is_ascii_digit())
            && (i == 0 || !is_flag_char(bytes[i - 1]));
        if !opens {
            i += 1;
            continue;
        }
        let start = i;
        i += 2;
        while i < bytes.len()
            && (bytes[i].is_ascii_lowercase() || bytes[i].is_ascii_digit() || bytes[i] == b'-')
        {
            i += 1;
        }
        let flag = text[start..i].trim_end_matches('-').to_string();
        if !BUILTIN_FLAGS.contains(&flag.as_str()) && !flags.contains(&flag) {
            flags.push(flag);
        }
    }
    flags
}

/// Long flags of one option-list line, reading only the flag-spec column so a
/// description mentioning another flag is ignored.
fn long_flags_in(line: &str) -> Vec<String> {
    scan_flags(line.trim_start().split("  ").next().unwrap_or_default())
}

fn is_flag_char(b: u8) -> bool {
    b == b'-' || b.is_ascii_lowercase() || b.is_ascii_digit()
}

/// Long flags advertised in one command's help, across every option heading.
fn flags_in(help: &str) -> Vec<String> {
    let mut flags: Vec<String> = Vec::new();
    for heading in headings(help) {
        for line in section(help, heading) {
            if !is_option_line(line) {
                continue;
            }
            for flag in long_flags_in(line) {
                if !flags.contains(&flag) {
                    flags.push(flag);
                }
            }
        }
    }
    flags
}

/// Flags a command's help names in PROSE — inside `back-ticks` — rather than
/// advertising in its option column. Two real shapes today:
///   * hidden clap aliases, which clap does not render in the spec column
///     (`stt-generate gtfs -o, --output …  "`--out` is accepted as an alias"`);
///   * another tool's flag quoted in a build note (`"requires `--features
///     postgres`"`), which the reference doc quotes the same way.
///
/// Used only to excuse doc mentions in `documented_flags_all_exist`; never
/// folded into `flags_in`. Clap never backticks the spec column, so this set is
/// disjoint from what the option-line parser produces — which is what keeps the
/// reverse gate usable as a backstop for that parser collapsing.
fn prose_flags_in(help: &str) -> Vec<String> {
    let mut flags = Vec::new();
    for span in inline_code_spans(help) {
        for flag in scan_flags(span) {
            if !flags.contains(&flag) {
                flags.push(flag);
            }
        }
    }
    flags
}

/// Subcommands advertised in one command's help (clap's own `help` excluded).
fn subcommands_in(help: &str) -> Vec<String> {
    section(help, "Commands:")
        .into_iter()
        .filter(|line| line.len() - line.trim_start().len() <= 4)
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| *name != "help" && name.starts_with(|c: char| c.is_ascii_lowercase()))
        .map(str::to_string)
        .collect()
}

/// Every command `exe` exposes — itself, then each advertised subcommand,
/// depth-first to `MAX_SUBCOMMAND_DEPTH` — paired with its `--help` text.
/// Help is fetched once per command and reused by both directions.
fn commands_of(exe: &str) -> Vec<(Vec<String>, String)> {
    fn walk(exe: &str, prefix: Vec<String>, depth: usize, out: &mut Vec<(Vec<String>, String)>) {
        let args: Vec<&str> = prefix.iter().map(String::as_str).collect();
        let help = help_text(exe, &args);
        let subs = if depth == 0 {
            Vec::new()
        } else {
            subcommands_in(&help)
        };
        out.push((prefix.clone(), help));
        for sub in subs {
            let mut next = prefix.clone();
            next.push(sub);
            walk(exe, next, depth - 1, out);
        }
    }
    let mut out = Vec::new();
    walk(exe, Vec::new(), MAX_SUBCOMMAND_DEPTH, &mut out);
    out
}

/// The contents of each `` `…` `` span in `text` (a Markdown line, or a whole
/// `--help` blob — clap wraps descriptions, so a span can straddle newlines).
fn inline_code_spans(text: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('`') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('`') else { break };
        spans.push(&after[..close]);
        rest = &after[close + 1..];
    }
    spans
}

fn first_word(text: &str) -> Option<&str> {
    text.split_whitespace().next()
}

/// Long flags the doc presents as OURS, each with the 1-based line it first
/// appears on.
///
/// Only two shapes count, which is what keeps another tool's flags out of the
/// set without an allow-list:
///   * a fenced code line (plus its `\`-continuations) whose first word is one
///     of `CLI_NAMES` — so `stt-build --input …` contributes, `cargo build
///     --release …` does not;
///   * an inline span that *is* a flag — `` `--publish` ``, `` `--output-dir
///     <DIR>` `` — or that starts with one of our CLI names.
///
/// Flags written in bare prose (no backticks) are ignored: the doc's own style
/// is to backtick every flag, and reading unquoted text would drag in prose
/// dashes.
fn documented_flags(doc: &str) -> Vec<(String, usize)> {
    let mut found: Vec<(String, usize)> = Vec::new();
    let mut in_fence = false;
    let mut harvest = false;
    let mut continued = false;

    for (idx, line) in doc.lines().enumerate() {
        let line_no = idx + 1;
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            harvest = false;
            continued = false;
            continue;
        }

        let mut hits: Vec<String> = Vec::new();
        if in_fence {
            if !continued {
                harvest = first_word(line).is_some_and(|w| CLI_NAMES.contains(&w));
            }
            if harvest {
                hits = scan_flags(line);
            }
            continued = line.trim_end().ends_with('\\');
        } else {
            for span in inline_code_spans(line) {
                let ours = span.trim_start().starts_with("--")
                    || first_word(span).is_some_and(|w| CLI_NAMES.contains(&w));
                if ours {
                    hits.extend(scan_flags(span));
                }
            }
        }

        for flag in hits {
            if !found.iter().any(|(seen, _)| *seen == flag) {
                found.push((flag, line_no));
            }
        }
    }
    found
}

/// Whether the doc has a table row whose FIRST cell is exactly `` `name` `` —
/// i.e. the row that documents `name`, not merely a backticked mention of that
/// word somewhere else in the file (`all` in particular is far too generic for
/// a bare substring match to mean anything).
fn has_table_row(doc: &str, name: &str) -> bool {
    let cell = format!("`{name}`");
    doc.lines().any(|line| {
        let Some(rest) = line.trim_start().strip_prefix('|') else {
            return false;
        };
        rest.split('|').next().is_some_and(|f| f.trim() == cell)
    })
}

/// Whether `flag` appears in the doc as a whole word — so a missing
/// `--min-zoom` cannot be masked by a documented `--min-zoom-field`.
fn documented(doc: &str, flag: &str) -> bool {
    let bytes = doc.as_bytes();
    let mut from = 0;
    while let Some(offset) = doc[from..].find(flag) {
        let start = from + offset;
        let end = start + flag.len();
        let before_ok = start == 0 || !is_flag_char(bytes[start - 1]);
        let after_ok = end == bytes.len() || !is_flag_char(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        // `flag` is ASCII, so `start + 1` is a char boundary.
        from = start + 1;
    }
    false
}

/// Assert every long flag of `exe` — top level and its whole subcommand tree —
/// is documented. `cli` is the user-facing name used in the failure message.
fn assert_documented(cli: &str, exe: &str) {
    let doc = reference_doc();
    let commands = commands_of(exe);

    let mut missing = Vec::new();
    let mut checked = 0usize;
    for (command, help) in &commands {
        for flag in flags_in(help) {
            checked += 1;
            if !documented(&doc, &flag) {
                let where_ = if command.is_empty() {
                    format!("`{cli}` (top level)")
                } else {
                    format!("`{cli} {}`", command.join(" "))
                };
                missing.push(format!("  {where_}: {flag}"));
            }
        }
    }

    assert!(
        checked > 0,
        "parsed zero flags out of `{cli} --help` — the help-output parser in \
         tests/cli_reference_doc.rs has stopped matching clap's format"
    );
    // Visible under `--nocapture`; a sudden collapse in the count is the tell
    // that the parser, not the doc, is what broke. A *partial* collapse (one
    // binary or one subcommand stops parsing) survives the `checked > 0` assert
    // above but not `documented_flags_all_exist`, which turns the vanished
    // flags into phantom doc mentions.
    eprintln!(
        "{cli}: checked {checked} flag(s) over {} command(s)",
        commands.len()
    );
    assert!(
        missing.is_empty(),
        "{} CLI flag(s) are missing from docs/api/cli-reference.md:\n{}\n\
         Document each one in docs/api/cli-reference.md (in the section for the \
         listed CLI/subcommand), then re-run this test.",
        missing.len(),
        missing.join("\n")
    );
}

#[cfg(feature = "build-cli")]
#[test]
fn stt_build_flags_are_documented() {
    assert_documented("stt-build", env!("CARGO_BIN_EXE_stt-build"));
}

#[cfg(feature = "optimize-cli")]
#[test]
fn stt_optimize_flags_are_documented() {
    // Covers analyze / recommend / inspect / diff / doctor / order-audit —
    // discovered from `--help`, so a new subcommand is gated automatically.
    assert_documented("stt-optimize", env!("CARGO_BIN_EXE_stt-optimize"));
}

#[cfg(feature = "validate-cli")]
#[test]
fn stt_validate_flags_are_documented() {
    assert_documented("stt-validate", env!("CARGO_BIN_EXE_stt-validate"));
}

#[cfg(feature = "bundle-cli")]
#[test]
fn stt_bundle_flags_are_documented() {
    // Covers pack / unpack.
    assert_documented("stt-bundle", env!("CARGO_BIN_EXE_stt-bundle"));
}

#[cfg(feature = "serve-core")]
#[test]
fn stt_serve_flags_are_documented() {
    assert_documented("stt-serve", env!("CARGO_BIN_EXE_stt-serve"));
}

/// `stt-generate` is checked at the SUBCOMMAND level, not the flag level.
///
/// Two reasons. First, it lives in its own workspace at `tools/stt-generate`,
/// so Cargo gives this test no `CARGO_BIN_EXE_stt-generate` — it is resolved as
/// a sibling of the test binary (`target/<profile>/stt-generate`), which no
/// root-workspace build produces any more; absent that, this skips loudly
/// rather than pretending the surface was covered (see the module docs for how
/// to put it there, and the stale-binary warning). Second, the reference doc
/// deliberately documents the
/// per-dataset flags by reference ("run `stt-generate <subcommand> --help`")
/// and enumerates the *subcommands* in a table, so the table is what there is
/// to pin: adding a dataset generator must add its row.
#[test]
fn stt_generate_subcommands_are_documented() {
    let Some(exe) = sibling_binary("stt-generate") else {
        eprintln!(
            "SKIPPED stt_generate_subcommands_are_documented: no \
             target/<profile>/stt-generate (it lives in its own workspace at \
             tools/stt-generate, so no CARGO_BIN_EXE_ is available here and no \
             root-workspace build produces it). Build it there and copy the \
             binary into the root target dir to include it."
        );
        return;
    };
    let exe = exe.to_str().expect("utf-8 target path");
    let doc = reference_doc();

    let subcommands = subcommands_in(&help_text(exe, &[]));
    assert!(
        !subcommands.is_empty(),
        "parsed zero subcommands out of `stt-generate --help` — the help-output \
         parser in tests/cli_reference_doc.rs has stopped matching clap's format"
    );

    // Each subcommand must own a row in the Subcommands table — matched on the
    // row's FIRST cell, not a bare backticked mention anywhere in the file, so
    // generic names (`all`) cannot be satisfied by unrelated prose.
    let missing: Vec<String> = subcommands
        .iter()
        .filter(|name| !has_table_row(&doc, name))
        .map(|name| format!("  `stt-generate {name}`"))
        .collect();
    assert!(
        missing.is_empty(),
        "{} stt-generate subcommand(s) are missing from \
         docs/api/cli-reference.md:\n{}\n\
         Add a row for each to the Subcommands table in the `stt-generate` \
         section of docs/api/cli-reference.md, then re-run this test.",
        missing.len(),
        missing.join("\n")
    );
}

/// The reverse gate: every long flag the doc writes in `code` must be a flag
/// one of the CLIs really has.
///
/// This is the direction the per-binary tests structurally cannot cover — they
/// only ever ask "is this real flag written down?", never "is this written-down
/// flag real?" — so without it a renamed or deleted flag survives in the doc
/// indefinitely.
///
/// Needs the WHOLE surface to be present, or a flag belonging to an absent
/// binary would read as a phantom: hence the all-five-features gate plus the
/// loud skip when `stt-generate` has not been built. For the same reason, the
/// one plausible false positive is a flag that only exists behind a non-default
/// cargo feature — the failure message says so.
#[cfg(all(
    feature = "build-cli",
    feature = "optimize-cli",
    feature = "validate-cli",
    feature = "bundle-cli",
    feature = "serve-core"
))]
#[test]
fn documented_flags_all_exist() {
    let Some(generate) = sibling_binary("stt-generate") else {
        eprintln!(
            "SKIPPED documented_flags_all_exist: no target/<profile>/stt-generate, \
             so the doc's per-dataset flags would read as phantoms. Build it from \
             tools/stt-generate and copy the binary into the root target dir to \
             include it."
        );
        return;
    };
    let exes: [(&str, &str); 6] = [
        ("stt-build", env!("CARGO_BIN_EXE_stt-build")),
        ("stt-optimize", env!("CARGO_BIN_EXE_stt-optimize")),
        ("stt-validate", env!("CARGO_BIN_EXE_stt-validate")),
        ("stt-bundle", env!("CARGO_BIN_EXE_stt-bundle")),
        ("stt-serve", env!("CARGO_BIN_EXE_stt-serve")),
        (
            "stt-generate",
            generate.to_str().expect("utf-8 target path"),
        ),
    ];

    let mut real: Vec<String> = Vec::new();
    let mut prose: Vec<String> = Vec::new();
    for (cli, exe) in exes {
        let mut own = 0usize;
        for (_, help) in commands_of(exe) {
            for flag in flags_in(&help) {
                own += 1;
                if !real.contains(&flag) {
                    real.push(flag);
                }
            }
            for flag in prose_flags_in(&help) {
                if !prose.contains(&flag) {
                    prose.push(flag);
                }
            }
        }
        // Every one of our CLIs advertises at least one non-builtin long flag,
        // so a zero here is a parser collapse for THIS binary — which flag-set
        // overlap between binaries would otherwise partly mask below.
        assert!(
            own > 0,
            "parsed zero flags out of `{cli} --help` — the help-output parser in \
             tests/cli_reference_doc.rs has stopped matching clap's format"
        );
    }

    let doc = reference_doc();
    let mentioned = documented_flags(&doc);
    assert!(
        !mentioned.is_empty(),
        "found zero backticked flags in docs/api/cli-reference.md — the Markdown \
         scanner in tests/cli_reference_doc.rs has stopped matching the doc"
    );

    let phantom: Vec<String> = mentioned
        .iter()
        .filter(|(flag, _)| !real.contains(flag) && !prose.contains(flag))
        .map(|(flag, line)| format!("  docs/api/cli-reference.md:{line}: {flag}"))
        .collect();
    eprintln!(
        "cli-reference: {} documented flag(s) checked against {} advertised + {} \
         prose-named flag(s)",
        mentioned.len(),
        real.len(),
        prose.len()
    );
    assert!(
        phantom.is_empty(),
        "{} flag(s) documented in docs/api/cli-reference.md are on no CLI — \
         neither advertised by clap nor named in any `--help` prose:\n{}\n\
         Either the flag was renamed/removed (fix the doc) or it lives behind a \
         non-default cargo feature that this build does not enable (in which \
         case build with that feature, or stop documenting it as unconditional).",
        phantom.len(),
        phantom.join("\n")
    );
}

/// `…/target/<profile>/deps/<test-bin>` → `…/target/<profile>/<name>`.
fn sibling_binary(name: &str) -> Option<PathBuf> {
    let test_exe = std::env::current_exe().ok()?;
    let profile_dir = test_exe.parent()?.parent()?;
    let candidate = profile_dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    candidate.is_file().then_some(candidate)
}
