// Fused AES-128-CTR + SHA-256 batch benchmark: CPU (OpenSSL, AES-NI/SHA-NI) vs CUDA GPU.
//
// This models the real per-block crypto cost of a Freenet/Hyphanet splitfile block
// (new CHK format: AES-CTR over 32 KiB, then SHA-256 of the ciphertext to form the
// key). Doing BOTH ops in one GPU pass amortizes the PCIe transfer that made a
// single SHA-256 pass transfer-bound.
//
// CPU baseline uses OpenSSL's EVP (hardware AES-NI + SHA-NI), so the comparison is fair.
// We avoid needing openssl dev headers by declaring the few EVP prototypes we use.
//
// Build:
//   nvcc -O3 -allow-unsupported-compiler -ccbin g++-13 -Xcompiler -fopenmp \
//        -o gpgpu/blockcrypto_bench gpgpu/blockcrypto_bench.cu /usr/lib/x86_64-linux-gnu/libcrypto.so.3
// Run (needs the real GPU):
//   ./gpgpu/blockcrypto_bench [numBlocks] [blockBytes]

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

// ---- Minimal OpenSSL EVP prototypes (avoids needing libssl-dev headers) ----
extern "C" {
typedef struct evp_cipher_st EVP_CIPHER;
typedef struct evp_cipher_ctx_st EVP_CIPHER_CTX;
typedef struct evp_md_st EVP_MD;
typedef struct engine_st ENGINE;
const EVP_CIPHER *EVP_aes_128_ctr(void);
EVP_CIPHER_CTX *EVP_CIPHER_CTX_new(void);
void EVP_CIPHER_CTX_free(EVP_CIPHER_CTX *);
int EVP_EncryptInit_ex(EVP_CIPHER_CTX *, const EVP_CIPHER *, ENGINE *, const unsigned char *, const unsigned char *);
int EVP_EncryptUpdate(EVP_CIPHER_CTX *, unsigned char *, int *, const unsigned char *, int);
const EVP_MD *EVP_sha256(void);
int EVP_Digest(const void *, size_t, unsigned char *, unsigned int *, const EVP_MD *, ENGINE *);
}

// ===========================================================================
// SHA-256 (host+device)
// ===========================================================================
__host__ __device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n){ return (x>>n)|(x<<(32-n)); }

__constant__ uint32_t Kc[64];
static const uint32_t Kh[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__host__ __device__ __forceinline__
void sha_chunk(uint32_t h[8], const uint8_t *p, const uint32_t *K){
    uint32_t w[64];
    #pragma unroll
    for(int i=0;i<16;i++) w[i]=(uint32_t(p[i*4])<<24)|(uint32_t(p[i*4+1])<<16)|(uint32_t(p[i*4+2])<<8)|uint32_t(p[i*4+3]);
    #pragma unroll
    for(int i=16;i<64;i++){
        uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll
    for(int i=0;i<64;i++){
        uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        uint32_t t1=hh+S1+ch+K[i]+w[i];
        uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+maj;
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
}

// ===========================================================================
// AES-128 encryption via T-tables (host builds tables + round keys; device runs).
// ===========================================================================
static const uint8_t SBOX[256] = {
0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__constant__ uint32_t Te0c[256],Te1c[256],Te2c[256],Te3c[256],Te4c[256];
__constant__ uint32_t RKc[44];

static inline uint8_t xtime(uint8_t x){ return (uint8_t)((x<<1) ^ ((x>>7)*0x1b)); }
static inline uint32_t ror32(uint32_t v,int n){ return (v>>n)|(v<<(32-n)); }

static void buildTables(uint32_t Te0[256],uint32_t Te1[256],uint32_t Te2[256],uint32_t Te3[256],uint32_t Te4[256]){
    for(int x=0;x<256;x++){
        uint8_t s=SBOX[x]; uint8_t b2=xtime(s); uint8_t b3=b2^s;
        Te0[x]=((uint32_t)b2<<24)|((uint32_t)s<<16)|((uint32_t)s<<8)|b3;
        Te1[x]=ror32(Te0[x],8); Te2[x]=ror32(Te0[x],16); Te3[x]=ror32(Te0[x],24);
        Te4[x]=((uint32_t)s<<24)|((uint32_t)s<<16)|((uint32_t)s<<8)|s;
    }
}
static void expandKey128(const uint8_t key[16], uint32_t rk[44]){
    static const uint32_t rcon[10]={0x01000000,0x02000000,0x04000000,0x08000000,0x10000000,
                                    0x20000000,0x40000000,0x80000000,0x1b000000,0x36000000};
    for(int i=0;i<4;i++)
        rk[i]=((uint32_t)key[4*i]<<24)|((uint32_t)key[4*i+1]<<16)|((uint32_t)key[4*i+2]<<8)|key[4*i+3];
    for(int i=4;i<44;i++){
        uint32_t t=rk[i-1];
        if(i%4==0){
            t=((uint32_t)SBOX[(t>>16)&0xff]<<24)|((uint32_t)SBOX[(t>>8)&0xff]<<16)|((uint32_t)SBOX[t&0xff]<<8)|SBOX[(t>>24)&0xff];
            t^=rcon[i/4-1];
        }
        rk[i]=rk[i-4]^t;
    }
}

__device__ __forceinline__ void aesEncryptBlock(const uint32_t in[4], uint32_t out[4]){
    uint32_t s0=in[0]^RKc[0], s1=in[1]^RKc[1], s2=in[2]^RKc[2], s3=in[3]^RKc[3];
    uint32_t t0,t1,t2,t3;
    #pragma unroll
    for(int r=1;r<10;r++){
        t0=Te0c[s0>>24]^Te1c[(s1>>16)&0xff]^Te2c[(s2>>8)&0xff]^Te3c[s3&0xff]^RKc[4*r+0];
        t1=Te0c[s1>>24]^Te1c[(s2>>16)&0xff]^Te2c[(s3>>8)&0xff]^Te3c[s0&0xff]^RKc[4*r+1];
        t2=Te0c[s2>>24]^Te1c[(s3>>16)&0xff]^Te2c[(s0>>8)&0xff]^Te3c[s1&0xff]^RKc[4*r+2];
        t3=Te0c[s3>>24]^Te1c[(s0>>16)&0xff]^Te2c[(s1>>8)&0xff]^Te3c[s2&0xff]^RKc[4*r+3];
        s0=t0;s1=t1;s2=t2;s3=t3;
    }
    out[0]=(Te4c[s0>>24]&0xff000000)^(Te4c[(s1>>16)&0xff]&0x00ff0000)^(Te4c[(s2>>8)&0xff]&0x0000ff00)^(Te4c[s3&0xff]&0x000000ff)^RKc[40];
    out[1]=(Te4c[s1>>24]&0xff000000)^(Te4c[(s2>>16)&0xff]&0x00ff0000)^(Te4c[(s3>>8)&0xff]&0x0000ff00)^(Te4c[s0&0xff]&0x000000ff)^RKc[41];
    out[2]=(Te4c[s2>>24]&0xff000000)^(Te4c[(s3>>16)&0xff]&0x00ff0000)^(Te4c[(s0>>8)&0xff]&0x0000ff00)^(Te4c[s1&0xff]&0x000000ff)^RKc[42];
    out[3]=(Te4c[s3>>24]&0xff000000)^(Te4c[(s0>>16)&0xff]&0x00ff0000)^(Te4c[(s1>>8)&0xff]&0x0000ff00)^(Te4c[s2&0xff]&0x000000ff)^RKc[43];
}

// One thread = one Freenet block: AES-128-CTR encrypt then SHA-256 of the ciphertext.
// IVs: 16 bytes per block. writeCipher: if non-null, store ciphertext (encode path).
__global__ void blockCryptoKernel(const uint8_t* __restrict__ plain, const uint8_t* __restrict__ ivs,
                                  uint32_t blockBytes, uint32_t numBlocks,
                                  uint8_t* __restrict__ cipherOut, uint8_t* __restrict__ digests){
    uint32_t idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=numBlocks) return;
    const uint8_t* p=plain+(size_t)idx*blockBytes;
    const uint8_t* iv=ivs+(size_t)idx*16;
    uint8_t* co = cipherOut ? cipherOut+(size_t)idx*blockBytes : nullptr;

    uint32_t iv0=(iv[0]<<24)|(iv[1]<<16)|(iv[2]<<8)|iv[3];
    uint32_t iv1=(iv[4]<<24)|(iv[5]<<16)|(iv[6]<<8)|iv[7];
    uint32_t iv2=(iv[8]<<24)|(iv[9]<<16)|(iv[10]<<8)|iv[11];
    uint32_t iv3=(iv[12]<<24)|(iv[13]<<16)|(iv[14]<<8)|iv[15];

    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t buf[64];
    uint32_t nblk = blockBytes/16;          // 2048 for 32 KiB
    for(uint32_t b=0;b<nblk;b++){
        // counter = IV + b (128-bit big-endian, b small so carry only low words)
        uint64_t t=(uint64_t)iv3+b; uint32_t c3=(uint32_t)t; uint32_t carry=(uint32_t)(t>>32);
        uint64_t t2c=(uint64_t)iv2+carry; uint32_t c2=(uint32_t)t2c; carry=(uint32_t)(t2c>>32);
        uint64_t t1c=(uint64_t)iv1+carry; uint32_t c1=(uint32_t)t1c; carry=(uint32_t)(t1c>>32);
        uint32_t c0=iv0+carry;
        uint32_t ctr[4]={c0,c1,c2,c3}, ks[4];
        aesEncryptBlock(ctr,ks);
        const uint8_t* pb=p+b*16;
        uint32_t pw0=(pb[0]<<24)|(pb[1]<<16)|(pb[2]<<8)|pb[3];
        uint32_t pw1=(pb[4]<<24)|(pb[5]<<16)|(pb[6]<<8)|pb[7];
        uint32_t pw2=(pb[8]<<24)|(pb[9]<<16)|(pb[10]<<8)|pb[11];
        uint32_t pw3=(pb[12]<<24)|(pb[13]<<16)|(pb[14]<<8)|pb[15];
        uint32_t e0=pw0^ks[0],e1=pw1^ks[1],e2=pw2^ks[2],e3=pw3^ks[3];
        uint32_t bo=(b&3)*16;               // position within the 64-byte SHA buffer
        buf[bo+0]=e0>>24;buf[bo+1]=e0>>16;buf[bo+2]=e0>>8;buf[bo+3]=e0;
        buf[bo+4]=e1>>24;buf[bo+5]=e1>>16;buf[bo+6]=e1>>8;buf[bo+7]=e1;
        buf[bo+8]=e2>>24;buf[bo+9]=e2>>16;buf[bo+10]=e2>>8;buf[bo+11]=e2;
        buf[bo+12]=e3>>24;buf[bo+13]=e3>>16;buf[bo+14]=e3>>8;buf[bo+15]=e3;
        if(co){ for(int k=0;k<16;k++) co[b*16+k]=buf[bo+k]; }
        if((b&3)==3) sha_chunk(h,buf,Kc);   // every 4 AES blocks = 64 bytes
    }
    // Padding (blockBytes multiple of 64 -> exactly one extra chunk)
    uint8_t pad[64];
    pad[0]=0x80; for(int i=1;i<56;i++) pad[i]=0;
    uint64_t bits=(uint64_t)blockBytes*8;
    for(int i=0;i<8;i++) pad[56+i]=(uint8_t)(bits>>((7-i)*8));
    sha_chunk(h,pad,Kc);
    uint8_t* d=digests+(size_t)idx*32;
    for(int i=0;i<8;i++){ d[i*4]=h[i]>>24;d[i*4+1]=h[i]>>16;d[i*4+2]=h[i]>>8;d[i*4+3]=h[i]; }
}

// Shared-memory T-table variant: constant memory serializes on AES's data-dependent
// lookups, so cooperatively stage the tables into shared memory first.
__global__ void blockCryptoKernelShared(const uint8_t* __restrict__ plain, const uint8_t* __restrict__ ivs,
                                  uint32_t blockBytes, uint32_t numBlocks,
                                  uint8_t* __restrict__ cipherOut, uint8_t* __restrict__ digests){
    __shared__ uint32_t T0[256],T1[256],T2[256],T3[256],T4[256];
    for(uint32_t i=threadIdx.x;i<256;i+=blockDim.x){
        T0[i]=Te0c[i];T1[i]=Te1c[i];T2[i]=Te2c[i];T3[i]=Te3c[i];T4[i]=Te4c[i];
    }
    __syncthreads();
    uint32_t idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=numBlocks) return;
    const uint8_t* p=plain+(size_t)idx*blockBytes;
    const uint8_t* iv=ivs+(size_t)idx*16;
    uint8_t* co = cipherOut ? cipherOut+(size_t)idx*blockBytes : nullptr;
    uint32_t iv0=(iv[0]<<24)|(iv[1]<<16)|(iv[2]<<8)|iv[3];
    uint32_t iv1=(iv[4]<<24)|(iv[5]<<16)|(iv[6]<<8)|iv[7];
    uint32_t iv2=(iv[8]<<24)|(iv[9]<<16)|(iv[10]<<8)|iv[11];
    uint32_t iv3=(iv[12]<<24)|(iv[13]<<16)|(iv[14]<<8)|iv[15];
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t buf[64];
    uint32_t nblk=blockBytes/16;
    for(uint32_t b=0;b<nblk;b++){
        uint64_t t=(uint64_t)iv3+b; uint32_t c3=(uint32_t)t; uint32_t carry=(uint32_t)(t>>32);
        uint64_t t2c=(uint64_t)iv2+carry; uint32_t c2=(uint32_t)t2c; carry=(uint32_t)(t2c>>32);
        uint64_t t1c=(uint64_t)iv1+carry; uint32_t c1=(uint32_t)t1c; carry=(uint32_t)(t1c>>32);
        uint32_t c0=iv0+carry;
        uint32_t s0=c0^RKc[0],s1=c1^RKc[1],s2=c2^RKc[2],s3=c3^RKc[3],r0,r1,r2,r3;
        #pragma unroll
        for(int r=1;r<10;r++){
            r0=T0[s0>>24]^T1[(s1>>16)&0xff]^T2[(s2>>8)&0xff]^T3[s3&0xff]^RKc[4*r+0];
            r1=T0[s1>>24]^T1[(s2>>16)&0xff]^T2[(s3>>8)&0xff]^T3[s0&0xff]^RKc[4*r+1];
            r2=T0[s2>>24]^T1[(s3>>16)&0xff]^T2[(s0>>8)&0xff]^T3[s1&0xff]^RKc[4*r+2];
            r3=T0[s3>>24]^T1[(s0>>16)&0xff]^T2[(s1>>8)&0xff]^T3[s2&0xff]^RKc[4*r+3];
            s0=r0;s1=r1;s2=r2;s3=r3;
        }
        uint32_t ks0=(T4[s0>>24]&0xff000000)^(T4[(s1>>16)&0xff]&0x00ff0000)^(T4[(s2>>8)&0xff]&0x0000ff00)^(T4[s3&0xff]&0x000000ff)^RKc[40];
        uint32_t ks1=(T4[s1>>24]&0xff000000)^(T4[(s2>>16)&0xff]&0x00ff0000)^(T4[(s3>>8)&0xff]&0x0000ff00)^(T4[s0&0xff]&0x000000ff)^RKc[41];
        uint32_t ks2=(T4[s2>>24]&0xff000000)^(T4[(s3>>16)&0xff]&0x00ff0000)^(T4[(s0>>8)&0xff]&0x0000ff00)^(T4[s1&0xff]&0x000000ff)^RKc[42];
        uint32_t ks3=(T4[s3>>24]&0xff000000)^(T4[(s0>>16)&0xff]&0x00ff0000)^(T4[(s1>>8)&0xff]&0x0000ff00)^(T4[s2&0xff]&0x000000ff)^RKc[43];
        const uint8_t* pb=p+b*16;
        uint32_t e0=((pb[0]<<24)|(pb[1]<<16)|(pb[2]<<8)|pb[3])^ks0;
        uint32_t e1=((pb[4]<<24)|(pb[5]<<16)|(pb[6]<<8)|pb[7])^ks1;
        uint32_t e2=((pb[8]<<24)|(pb[9]<<16)|(pb[10]<<8)|pb[11])^ks2;
        uint32_t e3=((pb[12]<<24)|(pb[13]<<16)|(pb[14]<<8)|pb[15])^ks3;
        uint32_t bo=(b&3)*16;
        buf[bo+0]=e0>>24;buf[bo+1]=e0>>16;buf[bo+2]=e0>>8;buf[bo+3]=e0;
        buf[bo+4]=e1>>24;buf[bo+5]=e1>>16;buf[bo+6]=e1>>8;buf[bo+7]=e1;
        buf[bo+8]=e2>>24;buf[bo+9]=e2>>16;buf[bo+10]=e2>>8;buf[bo+11]=e2;
        buf[bo+12]=e3>>24;buf[bo+13]=e3>>16;buf[bo+14]=e3>>8;buf[bo+15]=e3;
        if(co){ for(int k=0;k<16;k++) co[b*16+k]=buf[bo+k]; }
        if((b&3)==3) sha_chunk(h,buf,Kc);
    }
    uint8_t pad[64]; pad[0]=0x80; for(int i=1;i<56;i++) pad[i]=0;
    uint64_t bits=(uint64_t)blockBytes*8;
    for(int i=0;i<8;i++) pad[56+i]=(uint8_t)(bits>>((7-i)*8));
    sha_chunk(h,pad,Kc);
    uint8_t* d=digests+(size_t)idx*32;
    for(int i=0;i<8;i++){ d[i*4]=h[i]>>24;d[i*4+1]=h[i]>>16;d[i*4+2]=h[i]>>8;d[i*4+3]=h[i]; }
}

// ===========================================================================
static double ms_since(std::chrono::high_resolution_clock::time_point t0){
    return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
}
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__);exit(1);}}while(0)

int main(int argc,char**argv){
    uint32_t numBlocks=(argc>1)?atoi(argv[1]):8192;
    uint32_t blockBytes=(argc>2)?atoi(argv[2]):32768;
    size_t total=(size_t)numBlocks*blockBytes;
    double MiB=total/(1024.0*1024.0);
    printf("Fused AES-128-CTR + SHA-256: %u blocks x %u bytes = %.1f MiB\n",numBlocks,blockBytes,MiB);

    uint8_t key[16]; for(int i=0;i<16;i++) key[i]=(uint8_t)(0x10+i);
    std::vector<uint8_t> plain(total), ivs((size_t)numBlocks*16);
    for(size_t i=0;i<total;i++) plain[i]=(uint8_t)(i*1103515245u+12345u);
    for(size_t i=0;i<ivs.size();i++) ivs[i]=(uint8_t)(i*69069u+1u);

    std::vector<uint8_t> cpuCipher(total), cpuDig((size_t)numBlocks*32);
    std::vector<uint8_t> gpuCipher(total), gpuDig((size_t)numBlocks*32);

    // ---- CPU OpenSSL (AES-NI + SHA-NI) ----
    auto cpuRun=[&](int threads,std::vector<uint8_t>&cipher,std::vector<uint8_t>&dig)->double{
        auto t0=std::chrono::high_resolution_clock::now();
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads)
        #endif
        for(int b=0;b<(int)numBlocks;b++){
            EVP_CIPHER_CTX* ctx=EVP_CIPHER_CTX_new();
            EVP_EncryptInit_ex(ctx,EVP_aes_128_ctr(),nullptr,key,ivs.data()+(size_t)b*16);
            int outl=0;
            EVP_EncryptUpdate(ctx,cipher.data()+(size_t)b*blockBytes,&outl,plain.data()+(size_t)b*blockBytes,(int)blockBytes);
            EVP_CIPHER_CTX_free(ctx);
            unsigned int dl=0;
            EVP_Digest(cipher.data()+(size_t)b*blockBytes,blockBytes,dig.data()+(size_t)b*32,&dl,EVP_sha256(),nullptr);
        }
        return ms_since(t0);
    };
    double ms1=cpuRun(1,cpuCipher,cpuDig);
    printf("CPU  1-thread (OpenSSL): %8.2f ms  %7.2f MiB/s\n",ms1,MiB/(ms1/1000.0));
#ifdef _OPENMP
    int maxth=omp_get_max_threads();
    double msN=cpuRun(maxth,cpuCipher,cpuDig);
    printf("CPU %2d-thread (OpenSSL): %8.2f ms  %7.2f MiB/s\n",maxth,msN,MiB/(msN/1000.0));
#endif

    // ---- GPU ----
    uint32_t Te0[256],Te1[256],Te2[256],Te3[256],Te4[256],rk[44];
    buildTables(Te0,Te1,Te2,Te3,Te4); expandKey128(key,rk);
    CK(cudaMemcpyToSymbol(Kc,Kh,sizeof(Kh)));
    CK(cudaMemcpyToSymbol(Te0c,Te0,sizeof(Te0))); CK(cudaMemcpyToSymbol(Te1c,Te1,sizeof(Te1)));
    CK(cudaMemcpyToSymbol(Te2c,Te2,sizeof(Te2))); CK(cudaMemcpyToSymbol(Te3c,Te3,sizeof(Te3)));
    CK(cudaMemcpyToSymbol(Te4c,Te4,sizeof(Te4))); CK(cudaMemcpyToSymbol(RKc,rk,sizeof(rk)));

    uint8_t *d_plain,*d_ivs,*d_cipher,*d_dig;
    CK(cudaMalloc(&d_plain,total)); CK(cudaMalloc(&d_ivs,ivs.size()));
    CK(cudaMalloc(&d_cipher,total)); CK(cudaMalloc(&d_dig,(size_t)numBlocks*32));
    CK(cudaMemcpy(d_ivs,ivs.data(),ivs.size(),cudaMemcpyHostToDevice));
    int th=256, bl=(numBlocks+th-1)/th;

    // warm up
    CK(cudaMemcpy(d_plain,plain.data(),total,cudaMemcpyHostToDevice));
    blockCryptoKernel<<<bl,th>>>(d_plain,d_ivs,blockBytes,numBlocks,d_cipher,d_dig);
    CK(cudaDeviceSynchronize());

    // encode path: H2D(plain) + kernel + D2H(cipher + digests)  [shared-memory kernel]
    {
        auto t0=std::chrono::high_resolution_clock::now();
        CK(cudaMemcpy(d_plain,plain.data(),total,cudaMemcpyHostToDevice));
        blockCryptoKernelShared<<<bl,th>>>(d_plain,d_ivs,blockBytes,numBlocks,d_cipher,d_dig);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(gpuCipher.data(),d_cipher,total,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gpuDig.data(),d_dig,(size_t)numBlocks*32,cudaMemcpyDeviceToHost));
        double ms=ms_since(t0);
        printf("GPU encode e2e         : %8.2f ms  %7.2f MiB/s  (H2D plain + kernel + D2H cipher+digest)\n",ms,MiB/(ms/1000.0));
    }
    // verify path: H2D(plain) + kernel + D2H(digests only)  [shared-memory kernel]
    {
        auto t0=std::chrono::high_resolution_clock::now();
        CK(cudaMemcpy(d_plain,plain.data(),total,cudaMemcpyHostToDevice));
        blockCryptoKernelShared<<<bl,th>>>(d_plain,d_ivs,blockBytes,numBlocks,nullptr,d_dig);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(gpuDig.data(),d_dig,(size_t)numBlocks*32,cudaMemcpyDeviceToHost));
        double ms=ms_since(t0);
        printf("GPU verify e2e         : %8.2f ms  %7.2f MiB/s  (H2D plain + kernel + D2H digest only)\n",ms,MiB/(ms/1000.0));
    }
    // kernel only (constant-memory tables)
    {
        cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        blockCryptoKernel<<<bl,th>>>(d_plain,d_ivs,blockBytes,numBlocks,d_cipher,d_dig);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
        printf("GPU kernel only  (const): %8.2f ms  %7.2f MiB/s  (compute only)\n",ms,MiB/(ms/1000.0));
    }
    // kernel only (shared-memory tables)
    {
        // warm up shared variant
        blockCryptoKernelShared<<<bl,th>>>(d_plain,d_ivs,blockBytes,numBlocks,d_cipher,d_dig);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        blockCryptoKernelShared<<<bl,th>>>(d_plain,d_ivs,blockBytes,numBlocks,d_cipher,d_dig);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
        printf("GPU kernel only (shared): %8.2f ms  %7.2f MiB/s  (compute only)\n",ms,MiB/(ms/1000.0));
        CK(cudaMemcpy(gpuDig.data(),d_dig,(size_t)numBlocks*32,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gpuCipher.data(),d_cipher,total,cudaMemcpyDeviceToHost));
    }

    bool cOK = memcmp(cpuCipher.data(),gpuCipher.data(),total)==0;
    bool dOK = memcmp(cpuDig.data(),gpuDig.data(),(size_t)numBlocks*32)==0;
    printf("verify: ciphertext %s, digest %s vs OpenSSL\n", cOK?"OK":"FAIL", dOK?"OK":"FAIL");

    cudaFree(d_plain);cudaFree(d_ivs);cudaFree(d_cipher);cudaFree(d_dig);
    return (cOK&&dOK)?0:1;
}
