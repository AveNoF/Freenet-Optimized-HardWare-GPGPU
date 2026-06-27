# GPGPU experiment (CUDA) — research only, NOT wired into the node

This directory contains standalone CUDA micro-benchmarks that were used to evaluate
whether offloading Freenet's per-block cryptography (AES-CTR + SHA-256) to an NVIDIA
GPU is worthwhile. **The conclusion was that it is not**, so none of this code is
linked into the actual node. It is kept here only to document the experiment.

## Files
- `blockcrypto_bench.cu` — fused **AES-128-CTR + SHA-256** per 32 KiB block.
  CPU baseline uses **OpenSSL EVP** (so it benefits from AES-NI / SHA-NI), compared
  against two GPU kernels (constant-memory and shared-memory AES T-tables), measured
  both kernel-only and end-to-end (including PCIe host<->device transfer).
- `sha256_bench.cu` — batch **SHA-256** only, CPU (single + OpenMP) vs GPU.

## Build
```
nvcc -O3 -arch=native -Xcompiler -fopenmp \
     -allow-unsupported-compiler -ccbin g++-13 \
     blockcrypto_bench.cu -o blockcrypto_bench \
     -L/usr/lib/x86_64-linux-gnu -Xlinker -l:libcrypto.so.3
```
(Adjust `-ccbin` / library path to your system. The `-allow-unsupported-compiler`
flag is only needed when the host GCC is newer than what the CUDA toolkit supports.)

## What we found
- Raw GPU compute for the fused kernel (shared-memory T-tables) reached roughly
  **~2x** the throughput of a 16-thread CPU using AES-NI/SHA-NI.
- But once **PCIe transfer** is included, end-to-end GPU was **~0.5x** of CPU for the
  encode path (cipher+digest returned) and only **roughly on par** for the verify
  path (digest-only returned).

## Why we did NOT integrate it
1. The CPU already has AES-NI + SHA-NI; crypto is **not** the node's bottleneck
   (Freenet is overwhelmingly network/disk bound).
2. PCIe round-trips erase the raw compute win for the small 32 KiB blocks Freenet uses,
   unless huge batches are queued — which hurts latency.
3. It would add a hard CUDA dependency and a lot of complexity for, at best, a marginal
   and workload-specific gain.

The CPU-side optimizations that **were** kept (parallel splitfile encode/verify,
read-through datastore cache, HMAC instance reuse) live in the main source tree and are
measured by `test/freenet/bench/ForkBenchmarkTest.java`.
