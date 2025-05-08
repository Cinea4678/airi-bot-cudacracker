extern crate cc;

fn main() {
    cc::Build::new()
    .cuda(true)
    .flag("-cudart=shared")
    .flag("-gencode")
    .flag("arch=compute_61,code=sm_61")
    //.file("src/gpu_code/vector_add.cu")
    .file("src/gpu_code/md5.cu")
    .compile("libcudacracker.a");

/* Link CUDA Runtime (libcudart.so) */

// Add link directory
// - This path depends on where you install CUDA (i.e. depends on your Linux distribution)
// - This should be set by `$LIBRARY_PATH`
println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
println!("cargo:rustc-link-lib=cudart");

println!("cargo:rerun-if-changed=src/gpu_code/md5.cu");
}