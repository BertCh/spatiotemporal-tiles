// Migrate one real dataset dir in place and prove decode identity.
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: migrate-one <dataset-dir>");
    let dir = Path::new(&dir);
    let r = stt_core::pack::PackedReader::open(dir.join("manifest.json"))?;
    let before: Vec<_> = r
        .entries()
        .iter()
        .map(|e| {
            let layers = r.read_layers(e).unwrap();
            (
                e.zoom,
                e.x,
                e.y,
                e.time_start,
                layers
                    .iter()
                    .map(|l| (l.name.clone(), l.batch.num_rows()))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let fv_before = r.format_version();
    drop(r);
    match stt_core::pack::migrate_dataset_v2_to_v3(dir)? {
        None => println!("already v3, nothing to do"),
        Some(rep) => {
            let r2 = stt_core::pack::PackedReader::open(dir.join("manifest.json"))?;
            let after: Vec<_> = r2
                .entries()
                .iter()
                .map(|e| {
                    let layers = r2.read_layers(e).unwrap();
                    (
                        e.zoom,
                        e.x,
                        e.y,
                        e.time_start,
                        layers
                            .iter()
                            .map(|l| (l.name.clone(), l.batch.num_rows()))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            assert_eq!(r2.format_version(), 3);
            assert_eq!(before.len(), after.len(), "entry count moved");
            assert_eq!(before, after, "DECODED CONTENT DIFFERS");
            println!(
                "{}: v{} -> v3 | entries {} | packs untouched {} | paged {} | {} -> {} | decode IDENTICAL over all {} tiles",
                rep.dataset, fv_before, rep.entries, rep.packs_unchanged, rep.paged,
                rep.old_directory_key, rep.new_directory_key, after.len()
            );
        }
    }
    Ok(())
}
