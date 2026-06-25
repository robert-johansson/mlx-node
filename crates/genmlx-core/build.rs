fn main() {
    napi_build::setup();
    // mlx-core's #[napi] registration ctors survive the rlib->cdylib link with
    // no -force_load (spike genmlx-53lu, verified empirically). If a future
    // toolchain bump drops them (the downstream addon would be missing mlx-core
    // symbols like `zeros`/`MxArray`), force the linker to retain every object:
    //   println!("cargo:rustc-link-arg=-Wl,-force_load,<abs path to libmlx_core-*.rlib>");
}
