#include <stdio.h>
#include <time.h>

#define A0 (0x67452301)
#define B0 (0xefcdab89)
#define C0 (0x98badcfe)
#define D0 (0x10325476)
#define DIGEST_SIZE (16)
#define CHUNK_SIZE (64)
#define WORD_SIZE (4)
// How many MD5 hashes do we want to compute concurrently?
#define BATCH_SIZE (4194304)
#define CEIL(x) ((x) == (int)(x) ? (int)(x) : ((x) > 0 ? (int)(x) + 1 : (int)(x)))

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;

// Shift amounts in each MD5 round
__constant__ uint32_t shift_amts[64] = {
    7,
    12,
    17,
    22,
    7,
    12,
    17,
    22,
    7,
    12,
    17,
    22,
    7,
    12,
    17,
    22,
    5,
    9,
    14,
    20,
    5,
    9,
    14,
    20,
    5,
    9,
    14,
    20,
    5,
    9,
    14,
    20,
    4,
    11,
    16,
    23,
    4,
    11,
    16,
    23,
    4,
    11,
    16,
    23,
    4,
    11,
    16,
    23,
    6,
    10,
    15,
    21,
    6,
    10,
    15,
    21,
    6,
    10,
    15,
    21,
    6,
    10,
    15,
    21,
};
// Integer parts of signs of integers; used during the MD5 round operations
__constant__ uint32_t k_table[64] = {
    0xd76aa478,
    0xe8c7b756,
    0x242070db,
    0xc1bdceee,
    0xf57c0faf,
    0x4787c62a,
    0xa8304613,
    0xfd469501,
    0x698098d8,
    0x8b44f7af,
    0xffff5bb1,
    0x895cd7be,
    0x6b901122,
    0xfd987193,
    0xa679438e,
    0x49b40821,
    0xf61e2562,
    0xc040b340,
    0x265e5a51,
    0xe9b6c7aa,
    0xd62f105d,
    0x02441453,
    0xd8a1e681,
    0xe7d3fbc8,
    0x21e1cde6,
    0xc33707d6,
    0xf4d50d87,
    0x455a14ed,
    0xa9e3e905,
    0xfcefa3f8,
    0x676f02d9,
    0x8d2a4c8a,
    0xfffa3942,
    0x8771f681,
    0x6d9d6122,
    0xfde5380c,
    0xa4beea44,
    0x4bdecfa9,
    0xf6bb4b60,
    0xbebfbc70,
    0x289b7ec6,
    0xeaa127fa,
    0xd4ef3085,
    0x04881d05,
    0xd9d4d039,
    0xe6db99e5,
    0x1fa27cf8,
    0xc4ac5665,
    0xf4292244,
    0x432aff97,
    0xab9423a7,
    0xfc93a039,
    0x655b59c3,
    0x8f0ccc92,
    0xffeff47d,
    0x85845dd1,
    0x6fa87e4f,
    0xfe2ce6e0,
    0xa3014314,
    0x4e0811a1,
    0xf7537e82,
    0xbd3af235,
    0x2ad7d2bb,
    0xeb86d391,
};
// Context of MD5 computation
struct md5_ctx
{
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
};
// A vector of bytes; used to interface with the Rust code
struct FfiVector
{
    uint8_t *data;
    size_t len;
};
// A vector of FfiVectors; used to interface with the Rust code
struct FfiVectorBatch
{
    FfiVector *data;
    size_t len;
};

// Repeat initial context BATCH_SIZE times
md5_ctx init_ctxs[BATCH_SIZE];

// Left rotate 32-bit integer x by amt
__device__ uint32_t leftrotate(uint32_t x, uint32_t amt)
{
    return (x << (amt % 32)) | (x >> (32 - (amt % 32)));
}

__constant__ char d_prefix[64]; // 最长 63 字节；若你需要更长可调
__constant__ uint32_t d_target_a, d_target_b;

__device__ __forceinline__ void
write_length_le(uint8_t block[64], uint32_t bitlen)
{
    // little-endian 写入 64-bit 长度（仅低 32 位足够）
    reinterpret_cast<uint32_t *>(block + 56)[0] = bitlen;
}

__device__ __forceinline__ void swap(uint8_t &a, uint8_t &b)
{
    uint8_t tmp = a;
    a = b;
    b = tmp;
}

__global__ void
md5_kernel_find(uint64_t start_val, size_t prefix_len,
                int *match_flag, uint64_t *d_found_suffix)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid < 2)
    // {
    //     printf("Thread %d\n", tid);
    // }
    if (tid >= BATCH_SIZE)
        return;

    uint64_t num = start_val + tid;

    /* ---------- ① 生成 "prefix{num}" ---------- */
    uint8_t block[64] = {0};
    uint8_t len = prefix_len;

    // copy prefix
    for (int i = 0; i < prefix_len; ++i)
        block[i] = d_prefix[i];

    // 逆序写入十进制数字
    uint64_t tmp = num;
    int p = len;
    do
    {
        block[p++] = '0' + (tmp % 10);
        tmp /= 10;
    } while (tmp);
    // 反转数字区
    for (int i = 0; i < (p - len) / 2; ++i)
        swap(block[len + i], block[p - 1 - i]);
    len = p;

    /* ---------- ② MD5 单块预处理 ---------- */
    block[len] = 0x80;
    write_length_le(block, len * 8);

    /* ---------- ③ MD5 计算（64 步展开发生一次循环展开） ---------- */
    uint32_t a = A0, b = B0, c = C0, d = D0;
    const uint32_t *w = reinterpret_cast<uint32_t *>(block);

#pragma unroll
    for (int i = 0; i < 64; ++i)
    {
        uint32_t f, g;
        if (i < 16)
        {
            f = (b & c) | (~b & d);
            g = i;
        }
        else if (i < 32)
        {
            f = (d & b) | (~d & c);
            g = (5 * i + 1) & 15;
        }
        else if (i < 48)
        {
            f = b ^ c ^ d;
            g = (3 * i + 5) & 15;
        }
        else
        {
            f = c ^ (b | ~d);
            g = (7 * i) & 15;
        }

        f += a + k_table[i] + w[g];
        a = d;
        d = c;
        c = b;
        b += leftrotate(f, shift_amts[i]);
    }
    a += A0;
    b += B0;

    // if(num == 100) {
    //     printf("res: a: %08x, b: %08x, d_target_a: %08x, d_target_b: %08x\n", a, b, d_target_a, d_target_b);
    // }

    /* ---------- ④ 比对，仅首 6 字节 ---------- */
    if (a == d_target_a &&
        (b & 0xFFFF) == d_target_b)
    {
        if (atomicCAS(match_flag, -1, tid) == -1)
            d_found_suffix[0] = num;
    }
}

int md5_target_with_prefix(const char *h_prefix,
                           size_t prefix_len,
                           uint64_t start_value,
                           const uint8_t h_target_digest[DIGEST_SIZE],
                           uint64_t *h_found_suffix)
{
    /* ---- 一次性把 prefix / target 下到常量内存 ---- */
    cudaMemcpyToSymbol(d_prefix, h_prefix, prefix_len);
    uint32_t target_a = h_target_digest[0] |
                        (h_target_digest[1] << 8) |
                        (h_target_digest[2] << 16) |
                        (h_target_digest[3] << 24);
    uint32_t target_b = h_target_digest[4] |
                        (h_target_digest[5] << 8) |
                        (h_target_digest[6] << 16) |
                        (h_target_digest[7] << 24);
    printf("Target: %08x %08x; prefix: %s\n", target_a, target_b, h_prefix);

    cudaMemcpyToSymbol(d_target_a, &target_a, sizeof(uint32_t));
    cudaMemcpyToSymbol(d_target_b, &target_b, sizeof(uint32_t));

    /* ---- 设备侧返回值 ---- */
    int *d_flag; // -1 = 未找到
    uint64_t *d_suffix;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMalloc(&d_suffix, sizeof(uint64_t));

    const int TPB = 512;
    const int GRID = CEIL((float)BATCH_SIZE / TPB);

    uint64_t counter = 0;
    uint64_t last_value = start_value;
    uint64_t last_time = time(NULL);

    while (true)
    {
        int h_flag = -1;
        cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

        md5_kernel_find<<<GRID, TPB>>>(start_value, prefix_len,
                                       d_flag, d_suffix);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_flag >= 0)
        {
            cudaMemcpy(h_found_suffix, d_suffix,
                       sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaFree(d_flag);
            cudaFree(d_suffix);
            return 1; // 找到匹配
        }

        start_value += BATCH_SIZE; // 下一批
        /* 若需设置搜索上限，可在此处 break */

        if (++counter % 100 == 0)
        {
            uint64_t now = time(NULL);
            if (now == last_time)
                continue;

            float speed = (start_value - last_value) / float(now - last_time);
            printf("Searching... %llu (%f/s)\n",
                   start_value, speed);
            last_value = start_value;
            last_time = now;
        }
    }

    /* 不会到达 */
    cudaFree(d_flag);
    cudaFree(d_suffix);
    return 0;
}

extern "C"
{
    void init()
    {
        // Initialize the batched init_ctx
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            init_ctxs[i].a = A0;
            init_ctxs[i].b = B0;
            init_ctxs[i].c = C0;
            init_ctxs[i].d = D0;
        }
    }

    /**
     * C/Rust 外部接口：从 prefix + start_value 开始搜索与目标 MD5 匹配的字符串。
     *
     * @param prefix          指向 ASCII 字符串前缀的指针（不需要 null 结尾）
     * @param prefix_len      前缀长度
     * @param start_value     起始数字（用于拼接 test_prefix + 数字）
     * @param target_digest   指向目标 MD5（16字节）的指针
     * @param found_suffix    若成功匹配，将写入找到的数字
     * @return                1 = 找到匹配，0 = 未找到（理应不会触发），-1 = 异常
     */
    int md5_target_with_prefix_wrapper(const char *prefix,
                                       size_t prefix_len,
                                       uint64_t start_value,
                                       const uint8_t *target_digest,
                                       uint64_t *found_suffix)
    {
        if (!prefix || !target_digest || !found_suffix || prefix_len == 0)
        {
            return -1;
        }

        return md5_target_with_prefix(prefix, prefix_len, start_value, target_digest, found_suffix);
    }
}
