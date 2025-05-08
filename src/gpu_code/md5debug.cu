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
#define BATCH_SIZE (131072)
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

// Preprocess a batch of messages
__global__ void md5_preprocess_batched(uint8_t *pre_processed_msgs, size_t *pre_processed_sizes, size_t *orig_sizes, size_t *culmn_sizes)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Bounds-checking
    if (idx < BATCH_SIZE)
    {
        int n = orig_sizes[idx];
        int size_in_bits = 8 * n;
        int pre_processed_size = pre_processed_sizes[idx];

        // Add 0x80 byte
        pre_processed_msgs[culmn_sizes[idx] + n] = 0x80;
        // Adding the length
        for (int i = pre_processed_size - 8; i < pre_processed_size; i++)
        {
            int offset = i - (pre_processed_size - 8);
            pre_processed_msgs[culmn_sizes[idx] + (pre_processed_size - 8) + ((pre_processed_size - i) - 1)] = (size_in_bits >> ((7 - offset) * 8)) & 0xff;
        }
    }
}

// Modifying the contexts; this is the core of the MD5 computation
__global__ void md5_compute_batched(md5_ctx *ctxs, uint8_t *pre_processed_msgs, size_t *pre_processed_sizes, size_t *culmn_sizes)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Bounds-checking
    if (idx < BATCH_SIZE)
    {
        int pre_processed_size = pre_processed_sizes[idx];
        // Index using culminative sizes
        uint8_t *pre_processed_msg = pre_processed_msgs + culmn_sizes[idx];
        md5_ctx ctx = ctxs[idx];

        // Iterate over 64-byte chunks of the pre-processed message
        for (uint8_t *chunk = pre_processed_msg; chunk < pre_processed_msg + pre_processed_size; chunk += CHUNK_SIZE)
        {
            uint32_t words[CHUNK_SIZE / WORD_SIZE] = {0};

            // Break up the current chunk into words
            for (int word_idx = 0; word_idx < CHUNK_SIZE; word_idx += WORD_SIZE)
            {
                words[word_idx / WORD_SIZE] = chunk[word_idx] +
                                              (chunk[word_idx + 1] << 8) +
                                              (chunk[word_idx + 2] << 16) +
                                              (chunk[word_idx + 3] << 24);
            }

            // Start round
            uint32_t a = ctx.a;
            uint32_t b = ctx.b;
            uint32_t c = ctx.c;
            uint32_t d = ctx.d;

            // 64 round operations
            for (int i = 0; i < CHUNK_SIZE; i++)
            {
                uint32_t f;
                uint32_t g;

                if (i <= 15)
                {
                    f = ((b & c) | ((~b) & d));
                    g = i;
                }
                else if (16 <= i && i <= 31)
                {
                    f = ((d & b) | ((~d) & c));
                    g = (5 * i + 1) % 16;
                }
                else if (32 <= i && i <= 47)
                {
                    f = (b ^ c ^ d);
                    g = (3 * i + 5) % 16;
                }
                else
                {
                    f = (c ^ (b | (~d)));
                    g = (7 * i) % 16;
                }

                f = (f + a + k_table[i] + words[g]);
                a = d;
                d = c;
                c = b;
                b = (b + leftrotate(f, shift_amts[i]));
            }

            // Add to current registers
            ctxs[idx].a = (ctxs[idx].a + a);
            ctxs[idx].b = (ctxs[idx].b + b);
            ctxs[idx].c = (ctxs[idx].c + c);
            ctxs[idx].d = (ctxs[idx].d + d);
        }
    }
}

// Each thread compares its respective context from ctxs with the target_ctx
// and writes its index on the grid to match_idx if the contexts are identical
//
// Note: Airi bot only need the first 6 bytes of the digest to compare with the target digest.
__global__ void md5_compare_ctx_batched(md5_ctx *ctxs, md5_ctx *target_ctx, int *match_idx)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < BATCH_SIZE)
    {
        int start = idx * DIGEST_SIZE;
        md5_ctx ctx = ctxs[idx];

        // Compare first 6 bytes of the digest
        if (ctx.a == target_ctx->a &&
            (ctx.b & 0xFFFF) == (target_ctx->b & 0xFFFF))
        {
            *match_idx = idx;
        }
    }
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
    // printf("Thread %d\n", tid);
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
    cudaError_t err = cudaMemcpyToSymbol(d_prefix, h_prefix, prefix_len);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
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

        if (++counter % 5000 == 0)
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

// Attempt to find a message in the batch whose digest matches the target digest
int md5_target_batched(FfiVector *msgs, md5_ctx *h_target_ctx)
{
    uint8_t *h_pre_processed_msgs;
    uint8_t *d_pre_processed_msgs;
    size_t h_pre_processed_sizes[BATCH_SIZE];
    size_t *d_pre_processed_sizes;
    size_t h_orig_sizes[BATCH_SIZE];
    size_t *d_orig_sizes;
    size_t h_culmn_sizes[BATCH_SIZE];
    size_t *d_culmn_sizes;
    md5_ctx *h_ctxs[BATCH_SIZE * sizeof(md5_ctx)];
    md5_ctx *d_ctxs;
    md5_ctx *d_target_ctx;
    int *d_match_idx;
    int h_match_idx = -1;
    int total_size = 0;
    const int threads_per_block = 512;
    const int blocks_per_grid = CEIL((float)BATCH_SIZE / (float)threads_per_block);

    // Calculate the total size of the messages after pre-processing
    // We also fill in the size arrays, such as the size of each message after pre-processing
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        // Size of the i-th message after pre-processing
        // It is calculated as such because we have to fit in the message (msgs[i].len bytes), the message's length
        // as a 64-bit integer (8 bytes), and the additional 0x80 byte (1 byte)
        int pre_processed_size = CEIL((float)(msgs[i].len + 8 + 1) / (float)CHUNK_SIZE) * CHUNK_SIZE;
        h_pre_processed_sizes[i] = pre_processed_size;
        h_orig_sizes[i] = msgs[i].len;
        h_culmn_sizes[i] = (i == 0 ? 0 : h_culmn_sizes[i - 1] + pre_processed_size);
        total_size += pre_processed_size;
    }

    // Allocate enough memory for all of the pre-processed messages
    h_pre_processed_msgs = new uint8_t[total_size];
    // Memzeroing it eliminates the need for zero padding
    memset(h_pre_processed_msgs, 0, total_size);

    // Memcpy each message to its corresponding index
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        memcpy(h_pre_processed_msgs + h_culmn_sizes[i], msgs[i].data, msgs[i].len);
    }
    // Allocate space for the pre-processed messages on the device
    cudaMalloc(&d_pre_processed_msgs, total_size);
    // Array of culminative message sizes
    cudaMalloc(&d_culmn_sizes, sizeof(size_t) * BATCH_SIZE);
    // Array of pre-processed message sizes
    cudaMalloc(&d_pre_processed_sizes, sizeof(size_t) * BATCH_SIZE);
    // Array of original message sizes
    cudaMalloc(&d_orig_sizes, sizeof(size_t) * BATCH_SIZE);
    // Array of MD5 contexts
    cudaMalloc(&d_ctxs, sizeof(md5_ctx) * BATCH_SIZE);
    // The target context
    cudaMalloc(&d_target_ctx, sizeof(md5_ctx));
    // The integer threads write a match to (if one is found)
    cudaMalloc(&d_match_idx, sizeof(int));
    //  Memcpys to the variables allocated above
    cudaMemcpy(d_pre_processed_msgs, h_pre_processed_msgs, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_culmn_sizes, h_culmn_sizes, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pre_processed_sizes, h_pre_processed_sizes, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_orig_sizes, h_orig_sizes, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctxs, &init_ctxs, sizeof(md5_ctx) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_ctx, h_target_ctx, sizeof(md5_ctx), cudaMemcpyHostToDevice);
    cudaMemcpy(d_match_idx, &h_match_idx, sizeof(int), cudaMemcpyHostToDevice);

    // Preprocess the messages
    md5_preprocess_batched<<<blocks_per_grid, threads_per_block>>>(d_pre_processed_msgs, d_pre_processed_sizes, d_orig_sizes, d_culmn_sizes);
    cudaDeviceSynchronize();
    // Modify the states (the core of the MD5 computation)
    md5_compute_batched<<<blocks_per_grid, threads_per_block>>>(d_ctxs, d_pre_processed_msgs, d_pre_processed_sizes, d_culmn_sizes);
    cudaDeviceSynchronize();
    // Finalize: pack the contexts into digests
    md5_compare_ctx_batched<<<blocks_per_grid, threads_per_block>>>(d_ctxs, d_target_ctx, d_match_idx);
    // Copy into output match
    cudaMemcpy(&h_match_idx, d_match_idx, sizeof(int), cudaMemcpyDeviceToHost);
    // Free memory
    cudaFree(d_culmn_sizes);
    cudaFree(d_pre_processed_msgs);
    cudaFree(d_pre_processed_sizes);
    cudaFree(d_orig_sizes);
    cudaFree(d_ctxs);
    cudaFree(d_target_ctx);

    return h_match_idx;
}

/**
 * Rust 外部接口：从 prefix + start_value 开始搜索与目标 MD5 匹配的字符串。
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

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max shared memory per block: %lu\n", prop.sharedMemPerBlock);

    uint8_t target[16] = {104, 18, 21, 239, 203, 113};
    uint64_t found;
    int ok = md5_target_with_prefix("saki_", 6, 0, target, &found);
    if (ok)
        printf("Hit at suffix = %llu\n", found);
    else
        printf("Not found\n");
}