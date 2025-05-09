use std::{io, usize};

use clap::Parser;

#[link(name = "cudacracker", kind = "static")]
unsafe extern "C" {
    unsafe fn init();

    fn md5_target_with_prefix_wrapper(
        prefix: *const u8,
        prefix_len: usize,
        start_value: u64,
        target_digest: *const u8, // 长度必须为 16
        found_suffix: *mut u64,
    ) -> i32;
}

#[derive(Debug, Clone, clap::Parser)]
struct Arguments {
    target_digest_prefix: String,
    #[clap(short = 'p', default_value = "saki_")]
    test_prefix: String,
    #[clap(short = 's', default_value = "0")]
    start_point: usize,
}

fn main() -> Result<(), io::Error> {
    env_logger::init();

    unsafe {
        init();
    }

    let args = Arguments::parse();

    // 校验参数
    if args.target_digest_prefix.len() != 12 {
        panic!("目标摘要前缀长度必须为 12");
    }
    if args.test_prefix.len() > 16 {
        panic!("前缀长度不能超过 16");
    }

    let prefix = args.test_prefix.as_bytes();
    let target: [u8; 16] = {
        let target_raw: [u8; 6] = hex::decode(args.target_digest_prefix)
            .expect("目标前缀不是合法的hex字符串")
            .try_into()
            .expect("目标前缀长度错误");

        let mut target: [u8; 16] = [0; 16];
        (&mut target[0..6]).copy_from_slice(&target_raw);
        target
    };
    let mut found: u64 = 0;

    let result = unsafe {
        md5_target_with_prefix_wrapper(
            prefix.as_ptr(),
            prefix.len(),
            args.start_point as u64,
            target.as_ptr(),
            &mut found,
        )
    };

    if result == 1 {
        println!(
            "找到匹配: {}{}",
            std::str::from_utf8(prefix)
                .inspect_err(|_| {
                    println!("找到匹配，但内部错误！prefix = {prefix:?}");
                })
                .expect("找到匹配，但内部错误"),
            found
        );
    } else {
        println!("未找到匹配或出错");
    }

    Ok(())
}
