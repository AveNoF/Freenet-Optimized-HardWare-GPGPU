// Batch SHA-256 benchmark: CPU (single + OpenMP) vs CUDA GPU.
//
// This models the most universal crypto operation in Freenet/Hyphanet: hashing
// 32 KiB CHK blocks. Every CHK block is SHA-256'd on encode, decode and node
// verify. Blocks are independent, so the batch is embarrassingly parallel.
//
// The point of this microbenchmark is to decide *before* touching fred whether a
// GPU offload of block hashing actually beats using all CPU cores, once you
// include host<->device transfer over PCIe.
//
// Build:
//   nvcc -O3 -Xcompiler -fopenmp -o gpgpu/sha_bench gpgpu/sha256_bench.cu
// Run (needs the real GPU, i.e. outside any sandbox):
//   ./gpgpu/sha_bench [numBlocks] [blockBytes]

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// SHA-256 core, written so the exact same code runs on host and device.
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__constant__ uint32_t K_dev[64];
static const uint32_t K_host[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// Process one full 64-byte chunk into state h[8].
__host__ __device__ __forceinline__
void sha256_chunk(uint32_t h[8], const uint8_t* p, const uint32_t* K) {
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t(p[i*4]) << 24) | (uint32_t(p[i*4+1]) << 16) |
               (uint32_t(p[i*4+2]) << 8) | uint32_t(p[i*4+3]);
    }
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = hh + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

// Full SHA-256 of a message whose length is a multiple of 64 bytes (true for a
// 32 KiB block), with correct padding. out = 32 bytes.
__host__ __device__
void sha256(const uint8_t* msg, uint32_t len, uint8_t out[32], const uint32_t* K) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint32_t full = len & ~63u;
    for (uint32_t off = 0; off < full; off += 64)
        sha256_chunk(h, msg + off, K);

    // Padding into one or two final chunks.
    uint8_t buf[128];
    uint32_t rem = len - full;            // 0 for 32 KiB
    for (uint32_t i = 0; i < rem; i++) buf[i] = msg[full + i];
    uint32_t total = rem;
    buf[total++] = 0x80;
    uint32_t padTo = (rem < 56) ? 56 : 120;
    while (total < padTo) buf[total++] = 0x00;
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 7; i >= 0; i--) buf[total++] = (uint8_t)(bits >> (i*8));
    for (uint32_t off = 0; off < total; off += 64)
        sha256_chunk(h, buf + off, K);

    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(h[i] >> 24);
        out[i*4+1] = (uint8_t)(h[i] >> 16);
        out[i*4+2] = (uint8_t)(h[i] >> 8);
        out[i*4+3] = (uint8_t)(h[i]);
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: one thread per block.
// ---------------------------------------------------------------------------
__global__ void sha256_batch_kernel(const uint8_t* data, uint32_t blockBytes,
                                    uint32_t numBlocks, uint8_t* digests) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks) return;
    sha256(data + (size_t)idx * blockBytes, blockBytes, digests + (size_t)idx * 32, K_dev);
}

// ---------------------------------------------------------------------------

static double ms_since(std::chrono::high_resolution_clock::time_point t0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

int main(int argc, char** argv) {
    uint32_t numBlocks = (argc > 1) ? atoi(argv[1]) : 8192;     // 8192*32KiB = 256 MiB
    uint32_t blockBytes = (argc > 2) ? atoi(argv[2]) : 32768;
    size_t total = (size_t)numBlocks * blockBytes;
    double totalMiB = total / (1024.0*1024.0);

    printf("Batch SHA-256: %u blocks x %u bytes = %.1f MiB\n", numBlocks, blockBytes, totalMiB);

    // Self-test the SHA-256 against the known vector SHA256("abc").
    {
        uint8_t out[32];
        sha256((const uint8_t*)"abc", 3, out, K_host);
        const char* exp = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        char got[65];
        for (int i=0;i<32;i++) sprintf(got+i*2, "%02x", out[i]);
        printf("self-test SHA256(\"abc\") = %s [%s]\n", got, strcmp(got,exp)==0?"OK":"FAIL");
        if (strcmp(got,exp)!=0) return 1;
    }

    std::vector<uint8_t> h_data(total);
    for (size_t i = 0; i < total; i++) h_data[i] = (uint8_t)(i * 1103515245u + 12345u);
    std::vector<uint8_t> cpuDig((size_t)numBlocks*32), gpuDig((size_t)numBlocks*32);

    // ---- CPU single thread ----
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t b = 0; b < numBlocks; b++)
            sha256(h_data.data() + (size_t)b*blockBytes, blockBytes, cpuDig.data()+(size_t)b*32, K_host);
        double msc = ms_since(t0);
        printf("CPU  1-thread : %8.2f ms  %7.2f MiB/s\n", msc, totalMiB/(msc/1000.0));
    }

    // ---- CPU all cores (OpenMP) ----
#ifdef _OPENMP
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < (int)numBlocks; b++)
            sha256(h_data.data() + (size_t)b*blockBytes, blockBytes, gpuDig.data()+(size_t)b*32, K_host);
        double msc = ms_since(t0);
        printf("CPU %2d-thread : %8.2f ms  %7.2f MiB/s\n", omp_get_max_threads(), msc, totalMiB/(msc/1000.0));
    }
#endif

    // ---- GPU ----
    CK(cudaMemcpyToSymbol(K_dev, K_host, sizeof(K_host)));
    uint8_t *d_data=nullptr, *d_dig=nullptr;
    CK(cudaMalloc(&d_data, total));
    CK(cudaMalloc(&d_dig, (size_t)numBlocks*32));

    int threads = 256;
    int blocks = (numBlocks + threads - 1) / threads;

    // Warm up.
    CK(cudaMemcpy(d_data, h_data.data(), total, cudaMemcpyHostToDevice));
    sha256_batch_kernel<<<blocks, threads>>>(d_data, blockBytes, numBlocks, d_dig);
    CK(cudaDeviceSynchronize());

    // Timed: include H2D + kernel + D2H (the honest, end-to-end cost).
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CK(cudaMemcpy(d_data, h_data.data(), total, cudaMemcpyHostToDevice));
        sha256_batch_kernel<<<blocks, threads>>>(d_data, blockBytes, numBlocks, d_dig);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(gpuDig.data(), d_dig, (size_t)numBlocks*32, cudaMemcpyDeviceToHost));
        double msg = ms_since(t0);
        printf("GPU end2end  : %8.2f ms  %7.2f MiB/s  (H2D+kernel+D2H)\n", msg, totalMiB/(msg/1000.0));
    }
    // Timed: kernel only (steady-state if data already resident on GPU).
    {
        cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        sha256_batch_kernel<<<blocks, threads>>>(d_data, blockBytes, numBlocks, d_dig);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float msk=0; CK(cudaEventElapsedTime(&msk,a,b));
        printf("GPU kernel   : %8.2f ms  %7.2f MiB/s  (compute only)\n", msk, totalMiB/(msk/1000.0));
    }

    // Verify GPU == CPU.
    if (memcmp(cpuDig.data(), gpuDig.data(), (size_t)numBlocks*32) == 0)
        printf("verify: GPU digests match CPU  [OK]\n");
    else
        printf("verify: MISMATCH between GPU and CPU  [FAIL]\n");

    cudaFree(d_data); cudaFree(d_dig);
    return 0;
}
