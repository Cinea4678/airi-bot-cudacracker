use std::{
    env, fs, io::{self, Read}, slice, time, usize
};

use clap::Parser;
use itertools::Itertools;
use log::info;

// How many hashes do we compute at a time?
const BATCH_SIZE: usize = 16384;

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
    unsafe fn md5_target_batched_wrapper(msgs: &FfiVectorBatched, target_digest: &FfiVector)
        -> i32;
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

            data_slice
                .into_iter()
                .map(|x| slice::from_raw_parts(x.data, x.len).to_vec())
                .collect()
        }
    }
}

#[derive(Debug, Clone, clap::Parser)]
struct Arguments {
    target_digest_prefix: String,
    #[clap(short = 'p', default_value = "saki_")]
    test_prefix: String,
    #[clap(short = 's', default_value = "0")]
    start_point: usize,
}

// From the arguments, generates digests for the test strings and finds a string whose digest matches the input.
fn crack(args: Arguments) -> Option<String> {
    let mut current_point = args.start_point;
    let target_prefix = args.target_digest_prefix;
    let test_prefix = args.test_prefix;

    if target_prefix.len() != 12 {
        panic!("Target digest prefix must be 12 characters long");
    }
    let dec_digest = hex::decode(target_prefix).expect("Failed to decode digest");

    let mut last_point = (current_point, time::Instant::now());

    let mut wordlist = vec![String::new(); BATCH_SIZE * 4];

    loop {
        if current_point == usize::MAX {
            return None;
        }

        let next_batch_size = (usize::MAX - current_point).min(BATCH_SIZE * 4);
        (0..next_batch_size)
            .for_each(|i| {
                let idx = current_point + i;
                wordlist[i] = format!("{}{}", test_prefix, idx);
            });
        wordlist.truncate(next_batch_size);

        if let Some(answer) = crack_inner(&dec_digest, &wordlist) {
            return Some(answer);
        }
        current_point += next_batch_size;

        let current_time = time::Instant::now();
        let speed = (current_point - last_point.0) as f64 / (current_time - last_point.1).as_secs_f64();
        info!("Attempts so far: {}, speed: {}/sec", current_point - args.start_point, speed);

        last_point = (current_point, current_time);
    }
}

// From the wordlist, find a string whose digest matches the input; if such a string does not exist, return None
fn crack_inner(dec_digest: &[u8], wordlist: &[String]) -> Option<String> {
    let target_digest = FfiVector::from(dec_digest.to_owned());

    for chunk in wordlist.chunks(BATCH_SIZE) {
        let batch = FfiVectorBatched::from(
            chunk
                .into_iter()
                .map(|x| x.as_bytes().to_vec())
                .collect::<Vec<Vec<u8>>>(),
        );
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
    env_logger::init();

    unsafe {
        init();
    }

    let args = Arguments::parse();

    if let Some(result) = crack(args) {
        println!("Hash cracked: {result}");
    } else {
        println!("Couldn't crack hash");
    }

    Ok(())
}
