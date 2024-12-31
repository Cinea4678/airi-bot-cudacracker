use std::{
    env, fs, slice,
    io::{self, Read},
};

// How many hashes do we compute at a time?
const BATCH_SIZE: usize = 4096;

// Vector of bytes; used to interface w/CUDA code
#[repr(C)]
#[derive(Debug)]
pub struct FfiVector {
    data: *mut u8,
    len: usize,
}

// Vector of FfiVectors; used to interface w/CUDA code
#[repr(C)]
#[derive(Debug)]
pub struct FfiVectorBatched {
    data: *mut FfiVector,
    len: usize,
}

#[link(name = "cudacracker", kind = "static")]
unsafe extern "C" {
    unsafe fn init();
    unsafe fn md5_target_batched_wrapper(msgs: &FfiVectorBatched, target_digest: &FfiVector) -> i32;
}

impl From<Vec<u8>> for FfiVector {
    fn from(value: Vec<u8>) -> Self {
        let len = value.len();
        let data = value.as_ptr() as *mut u8;
        std::mem::forget(value);

        FfiVector { data, len }
    }
}

impl From<FfiVector> for Vec<u8> {
    fn from(value: FfiVector) -> Self {
        let n = value.len;
        let data = value.data;

        unsafe {
            let data_slice = slice::from_raw_parts(data, n);
            
            data_slice.to_vec()
        }
    }
}

impl From<Vec<FfiVector>> for FfiVectorBatched {
    fn from(value: Vec<FfiVector>) -> Self {
        let len = value.len();
        let data = value.as_ptr() as *mut FfiVector;
        std::mem::forget(value);

        FfiVectorBatched { data, len }
    }
}

impl From<Vec<Vec<u8>>> for FfiVectorBatched {
    fn from(value: Vec<Vec<u8>>) -> Self {
        let ffi_vecs: Vec<FfiVector> = value.into_iter().map(|x| FfiVector::from(x)).collect();
        
        FfiVectorBatched::from(ffi_vecs)
    }
}

impl From<FfiVectorBatched> for Vec<Vec<u8>> {
    fn from(value: FfiVectorBatched) -> Self {
        let n = value.len;
        let data = value.data;

        unsafe {
            let data_slice = slice::from_raw_parts(data, n);
            
            data_slice.into_iter().map(|x| slice::from_raw_parts(x.data, x.len).to_vec()).collect()
        }
    }
}

// From the wordlist, find a string whose digest matches the input; if such a string does not exist, return None
fn crack(digest: &str, wordlist: Vec<&str>) -> Option<String> {
    let dec_digest = hex::decode(digest).expect("Failed to decode digest");
    let target_digest = FfiVector::from(dec_digest);

    for chunk in wordlist.chunks(BATCH_SIZE) {
        let batch = FfiVectorBatched::from(chunk.into_iter().map(|x| x.as_bytes().to_vec()).collect::<Vec<Vec<u8>>>());
        unsafe {
            let idx = md5_target_batched_wrapper(&batch, &target_digest);

            // If the index is not -1, we have a match within the current batch
            if idx != -1 {
                return Some(chunk[idx as usize].to_string());
            }
        }
    }

    None
}

fn main() -> Result<(), io::Error> {
    unsafe {
        init();
    }
    
    let mut wordlist_file =
        fs::File::open(env::args().nth(1).expect("Expected wordlist file name"))?;
    let mut wordlist_data = String::new();
    let digest = env::args().nth(2).expect("Expected hash");

    wordlist_file.read_to_string(&mut wordlist_data)?;
    let wordlist = wordlist_data.lines().collect();

    if let Some(result) = crack(&digest, wordlist) {
        println!("Hash cracked: md5({result}) = {digest}");
    } else {
        println!("Couldn't crack hash");
    }

    Ok(())
}