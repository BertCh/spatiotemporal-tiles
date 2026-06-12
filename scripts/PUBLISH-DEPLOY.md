# Publish deploy runbook (zstd-19 + paged directory)

The deploy ships two **lossless** wins baked into the build: zstd level 19
(−8..27% on the wire, decode-free) and a paged directory (cold reader fetches
only the leaves a viewport touches). Coordinate quantization is **not** in this
deploy (it's lossy + per-dataset; opt in later via `--quantize-coords`).

These wins are in the **core build** now: the directory is **paged by default**
(opt out with `--single-directory`), `stt-build --publish` (which the
`stt-generate` workflow passes automatically) raises the level to 19, and the
`PackWriter` compresses blobs in parallel. The fleet migration below is just the
fast, lossless way to apply the same wins to the **already-built** datasets
without re-running every (often source-blocked) generator.

## 1. Build the fleet → staging  (done by `repack-publish-all.sh`)

```
cargo build --release -p stt-core --example repack-publish -p stt-validate
scripts/repack-publish-all.sh            # all datasets -> public/data-publish/
```

- Reads each **current packed dir** as the source of truth (several datasets
  were rebuilt later than their legacy `.stt`, so the packs carry the latest
  fixes), re-compresses losslessly to `public/data-publish/<ds>/`.
- Per-dataset: validates `decoded N of N` + tile-count preserved before staging;
  a bad build is dropped, the live tree is never touched. Resumable (skips
  already-staged); `FORCE=1` to rebuild.
- ~3 h on a 12-core machine (compute-bound, parallel). Watch: `tail -f /tmp/fleet-publish.log`.

## 2. Verify staging

```
du -sh examples/showcase/public/data-publish/*        # sizes vs live
target/release/stt-validate examples/showcase/public/data-publish/<ds>/manifest.json
```

Spot-check a few manifests carry `"layout":"paged"` + `"encoding":"zstd"`.

## 3. Ship the reader FIRST  (ordering matters)

The paged directory needs the paged-capable reader **live before** the new
(paged) manifests are served. The new reader also decodes the *old* single/zstd-3
data, so "new reader + old data on R2" is a safe intermediate state.

```
npx turbo run build --filter='./packages/*'   # @stt/core (+deck.gl/maplibre)
```

Deploy the showcase (Amplify builds it against these packages — see `amplify.yml`).
Confirm the live site loads current datasets before step 4.

## 4. Sync the new data to R2

Sync the staging tree straight to R2 — no destructive local swap (the live
`public/data/` stays as a rollback source):

```
STT_DATA_DIR=examples/showcase/public/data-publish scripts/r2-sync.sh --dry-run   # review
STT_DATA_DIR=examples/showcase/public/data-publish scripts/r2-sync.sh             # go
```

- Packs/index are content-addressed → new bytes get new names, uploaded
  alongside the old; only `manifest.json` (mutable, 60 s TTL) flips a dataset to
  the new packs. Old packs are GC'd retention-aware (default 7 d) after sessions
  drain — no cache purge needed.
- Per-dataset: `... r2-sync.sh <stem>` after re-staging just that one.

## 5. Promote staging → live (optional, after R2 verified)

Once the deploy is confirmed live, make the staging tree the new local source of
truth (keeps a backup):

```
for d in examples/showcase/public/data-publish/*/; do s=$(basename "$d")
  rm -rf "examples/showcase/public/data-old/$s"; mkdir -p examples/showcase/public/data-old
  mv "examples/showcase/public/data/$s" "examples/showcase/public/data-old/$s"
  mv "$d" "examples/showcase/public/data/$s"
done
```

## Rollback

R2 still holds the previous packs (until GC). Revert the showcase to the prior
reader and re-sync the old manifests (`scripts/r2-sync.sh` from the unswapped
`public/data/`, or restore from `data-old/`). No data is destroyed by the deploy.
