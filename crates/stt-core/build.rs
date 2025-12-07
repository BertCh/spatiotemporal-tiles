use prost_build::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    let proto_files = [
        "../../proto/tile.proto",
        "../../proto/index.proto",
        "../../proto/metadata.proto",
    ];

    let mut config = Config::new();
    config.type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    config
        .compile_protos(&proto_files, &["../../proto"])
        .expect("Failed to compile protocol buffers");

    // Tell Cargo to recompile if proto files change
    for file in &proto_files {
        println!("cargo:rerun-if-changed={}", file);
    }
}
