# Freenet-Optimized-HardWare-GPGPU

A performance-focused fork of [Hyphanet/Freenet (fred)](https://github.com/hyphanet/fred).

This fork keeps full protocol/wire compatibility with upstream Freenet but adds
CPU-side optimizations that make better use of modern multi-core CPUs, hardware
crypto (AES-NI / SHA-NI), and RAM. It also contains a documented GPGPU (CUDA)
research experiment that was evaluated and deliberately **not** wired into the node.

> Upstream documentation (build instructions, contributing, etc.) is preserved in
> [`README.upstream.md`](README.upstream.md).

---

## What's different from stock Freenet

| Area | Change | File(s) |
|------|--------|---------|
| **Splitfile FEC** | Block encode/verify (AES-CTR + SHA-256) is run in parallel across cores instead of serially. | `src/freenet/client/async/SplitFileFetcherSegmentStorage.java` |
| **Datastore** | `CachingFreenetStore` is now a **read-through** cache: blocks fetched from disk are kept in an LRU so repeated reads are served from RAM. Shared, configurable byte budget. | `src/freenet/store/caching/CachingFreenetStore.java`, `CachingFreenetStoreTracker.java` |
| **HMAC** | `Mac` instances are reused per-thread (`ThreadLocal`) instead of `Mac.getInstance()` on every packet. | `src/freenet/crypt/HMAC.java` |
| **Crypto visibility** | The selected JCE provider (e.g. SunJCE with AES-NI/SHA-NI) is now logged at startup. | `src/freenet/crypt/JceLoader.java` |

### Tuning

- `-Dfreenet.store.caching.readCacheSize=<bytes>` — RAM budget for the read-through
  cache (e.g. `134217728` for 128 MiB). `0` disables it (stock behaviour).
- `-Dfreenet.client.fec.encodeThreads=<n>` — worker threads for parallel splitfile
  encode/verify (defaults to the number of available processors).

---

## Measured before/after

Micro-benchmarks against stock behaviour using the **real** fred code, on a
16-thread CPU with AES-NI/SHA-NI enabled (see
[`test/freenet/bench/ForkBenchmarkTest.java`](test/freenet/bench/ForkBenchmarkTest.java)):

| Benchmark | Stock | This fork | Speedup |
|-----------|-------|-----------|---------|
| Splitfile block encode/verify (128 MiB) | 572 MiB/s (serial) | 4,752 MiB/s (16 cores) | **~8.3x** |
| HMAC-SHA256 per packet (300k ops) | 877k ops/s | 1,140k ops/s | **~1.3x** |
| Repeated CHK fetch, hot working set | 3,194 fetch/s | 1,108,408 fetch/s | **~347x** (cache hit) |

**Honest reading:** these are component-level numbers, not whole-node end-to-end
throughput. Freenet is usually network/disk bound, so the real-world benefit depends
on your workload — the splitfile and HMAC gains help CPU-bound nodes (many cores, fat
pipe, large downloads), and the read-cache gain only materializes on cache hits
(re-accessed / popular content). On a thin connection with no re-access, behaviour is
essentially unchanged.

Run them yourself:

```
./gradlew test --tests freenet.bench.ForkBenchmarkTest
```

(results are printed to the test's stdout)

---

## GPGPU experiment (research only)

The [`gpgpu/`](gpgpu/) directory contains standalone CUDA benchmarks that fuse
AES-128-CTR + SHA-256 per 32 KiB block and compare a GPU against an OpenSSL CPU
baseline (AES-NI/SHA-NI). **Conclusion: not worth integrating.** Raw GPU compute was
~2x a 16-thread CPU, but once PCIe transfer is included it was ~0.5x (encode) to about
on-par (verify) — and crypto isn't Freenet's bottleneck anyway. See
[`gpgpu/README.md`](gpgpu/README.md) for the full write-up.

---

## License

Same as upstream Freenet (GPLv2+). See [`README.upstream.md`](README.upstream.md) and
the `LICENSE*` files.
