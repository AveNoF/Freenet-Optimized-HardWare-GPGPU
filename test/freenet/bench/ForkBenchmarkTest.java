package freenet.bench;

import java.io.File;
import java.security.MessageDigest;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import freenet.crypt.HMAC;
import freenet.keys.CHKBlock;
import freenet.keys.ClientCHK;
import freenet.keys.ClientCHKBlock;
import freenet.keys.Key;
import freenet.node.SemiOrderedShutdownHook;
import freenet.store.CHKStore;
import freenet.store.caching.CachingFreenetStore;
import freenet.store.caching.CachingFreenetStoreTracker;
import freenet.store.saltedhash.ResizablePersistentIntBuffer;
import freenet.store.saltedhash.SaltedHashFreenetStore;
import freenet.support.PooledExecutor;
import freenet.support.Ticker;
import freenet.support.TrivialTicker;

/**
 * Before/after micro-benchmarks for the performance fork, run against the real fred code.
 * These are NOT correctness tests; they print throughput/latency to stdout so we can quote
 * concrete numbers for the optimizations vs the original ("stock") behaviour.
 *
 * Run: ./gradlew test --tests freenet.bench.ForkBenchmark
 */
public class ForkBenchmarkTest {

    @Rule
    public final TemporaryFolder temporaryFolder = new TemporaryFolder();

    private static double mibPerSec(long bytes, double ns) {
        return (bytes / (1024.0*1024.0)) / (ns / 1e9);
    }

    // ------------------------------------------------------------------
    // 1) Splitfile block encode/verify: serial (stock) vs parallel (fork)
    //    This is exactly the per-block AES-CTR + SHA-256 work that the fork
    //    spreads across cores during FEC decode.
    // ------------------------------------------------------------------
    @Test
    public void benchSplitfileEncode() throws Exception {
        final int N = 4096;                       // 4096 * 32KiB = 128 MiB per pass
        final int LEN = CHKBlock.DATA_LENGTH;     // 32768
        final long totalBytes = (long) N * LEN;
        Random r = new Random(1234);
        final byte[][] data = new byte[N][LEN];
        for (int i = 0; i < N; i++) r.nextBytes(data[i]);
        final byte[] cryptoKey = new byte[32];
        r.nextBytes(cryptoKey);
        final byte algo = Key.ALGO_AES_CTR_256_SHA256;

        // Warmup (both paths) to get the JIT hot.
        for (int w = 0; w < 2; w++) {
            encodeSerial(data, cryptoKey, algo);
            encodeParallel(data, cryptoKey, algo);
        }

        double bestSerial = Double.MAX_VALUE, bestPar = Double.MAX_VALUE;
        for (int it = 0; it < 3; it++) {
            long t0 = System.nanoTime();
            encodeSerial(data, cryptoKey, algo);
            bestSerial = Math.min(bestSerial, System.nanoTime() - t0);

            t0 = System.nanoTime();
            encodeParallel(data, cryptoKey, algo);
            bestPar = Math.min(bestPar, System.nanoTime() - t0);
        }

        int cores = Runtime.getRuntime().availableProcessors();
        System.out.println("\n=== [1] Splitfile block encode/verify (AES-CTR+SHA-256), "+N+" x 32KiB = 128 MiB ===");
        System.out.printf("  STOCK  (serial)        : %8.2f ms  %8.2f MiB/s%n", bestSerial/1e6, mibPerSec(totalBytes, bestSerial));
        System.out.printf("  FORK   (parallel, %2d c): %8.2f ms  %8.2f MiB/s%n", cores, bestPar/1e6, mibPerSec(totalBytes, bestPar));
        System.out.printf("  >>> speedup: %.2fx%n", bestSerial / bestPar);
    }

    private static void encodeSerial(byte[][] data, byte[] cryptoKey, byte algo) throws Exception {
        for (byte[] d : data)
            ClientCHKBlock.encodeSplitfileBlock(d, cryptoKey, algo);
    }

    private static void encodeParallel(byte[][] data, byte[] cryptoKey, byte algo) throws Exception {
        ForkJoinPool pool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
        try {
            pool.submit(() -> IntStream.range(0, data.length).parallel().forEach(i -> {
                try { ClientCHKBlock.encodeSplitfileBlock(data[i], cryptoKey, algo); }
                catch (Exception e) { throw new RuntimeException(e); }
            })).get();
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        } finally {
            pool.shutdown();
        }
    }

    // ------------------------------------------------------------------
    // 2) HMAC-SHA256: Mac.getInstance per call (stock) vs ThreadLocal reuse (fork)
    //    This runs once per network packet.
    // ------------------------------------------------------------------
    @Test
    public void benchHmac() throws Exception {
        final int ITERS = 300_000;
        final byte[] key = new byte[32];
        new Random(7).nextBytes(key);
        final byte[] msg = new byte[1400];        // ~one UDP packet
        new Random(8).nextBytes(msg);

        // Warmup
        for (int i = 0; i < 20_000; i++) { hmacStock(key, msg); HMAC.macWithSHA256(key, msg); }

        long t0 = System.nanoTime();
        for (int i = 0; i < ITERS; i++) hmacStock(key, msg);
        double stock = System.nanoTime() - t0;

        t0 = System.nanoTime();
        for (int i = 0; i < ITERS; i++) HMAC.macWithSHA256(key, msg);
        double fork = System.nanoTime() - t0;

        System.out.println("\n=== [2] HMAC-SHA256 per packet ("+ITERS+" ops, 1400-byte msg) ===");
        System.out.printf("  STOCK (getInstance/call): %8.2f ms  %10.0f ops/s%n", stock/1e6, ITERS/(stock/1e9));
        System.out.printf("  FORK  (ThreadLocal reuse): %8.2f ms  %10.0f ops/s%n", fork/1e6, ITERS/(fork/1e9));
        System.out.printf("  >>> speedup: %.2fx%n", stock / fork);
    }

    private static byte[] hmacStock(byte[] key, byte[] data) throws Exception {
        SecretKeySpec sk = new SecretKeySpec(key, "HmacSHA256");
        Mac mac = Mac.getInstance("HmacSHA256");   // the stock per-call cost
        mac.init(sk);
        return mac.doFinal(data);
    }

    // ------------------------------------------------------------------
    // 3) Read-through cache: repeated CHK fetch from disk store, cache OFF vs ON.
    // ------------------------------------------------------------------
    @Test
    public void benchReadThroughCache() throws Exception {
        ResizablePersistentIntBuffer.setPersistenceTime(-1);
        final int N = 800;                         // hot working set (~26 MiB), fits in cache
        final int REPEATS = 12;                    // how many times we re-fetch the working set
        PooledExecutor exec = new PooledExecutor();
        exec.start();
        Ticker ticker = new TrivialTicker(exec);

        // Note: Long.getLong needs a plain byte count (no "M" suffix).
        double off = runCacheScenario("0", N, REPEATS, ticker);            // read-through disabled (stock)
        double on  = runCacheScenario("134217728", N, REPEATS, ticker);    // 128 MiB read-through (fork)

        long fetches = (long) N * REPEATS;
        System.out.println("\n=== [3] Repeated CHK fetch: "+N+" blocks x "+REPEATS+" passes = "+fetches+" fetches ===");
        System.out.printf("  STOCK (read-cache OFF) : %8.2f ms  %10.0f fetches/s%n", off/1e6, fetches/(off/1e9));
        System.out.printf("  FORK  (read-cache ON)  : %8.2f ms  %10.0f fetches/s%n", on/1e6, fetches/(on/1e9));
        System.out.printf("  >>> speedup: %.2fx%n", off / on);
    }

    private double runCacheScenario(String readCacheSize, int N, int repeats, Ticker ticker) throws Exception {
        System.setProperty("freenet.store.caching.readCacheSize", readCacheSize);
        Random r = new Random(99);
        CHKStore store = new CHKStore();
        File f = new File(temporaryFolder.newFolder(), "bench_"+readCacheSize);
        try (SaltedHashFreenetStore<CHKBlock> saltStore = SaltedHashFreenetStore.construct(
                f, "benchCHK", store, r, N * 2, true, SemiOrderedShutdownHook.get(), true, true, ticker, null)) {
            CachingFreenetStoreTracker tracker = new CachingFreenetStoreTracker(4L*1024*1024, 300000, ticker);
            try (CachingFreenetStore<CHKBlock> cachingStore = new CachingFreenetStore<CHKBlock>(store, saltStore, tracker)) {
                cachingStore.start(null, true);

                // Write N distinct blocks straight to the on-disk store.
                ClientCHK[] keys = new ClientCHK[N];
                for (int i = 0; i < N; i++) {
                    byte[] data = new byte[CHKBlock.DATA_LENGTH];
                    r.nextBytes(data);
                    ClientCHKBlock b = ClientCHKBlock.encodeSplitfileBlock(data, null, Key.ALGO_AES_CTR_256_SHA256);
                    keys[i] = b.getClientKey();
                    saltStore.put(b.getBlock(), b.getBlock().getRawData(), b.getBlock().getRawHeaders(), false, false);
                }

                // Warmup pass (also populates the read cache in the ON scenario).
                for (int i = 0; i < N; i++)
                    store.fetch(keys[i].getNodeCHK(), false, false, null);

                long t0 = System.nanoTime();
                for (int pass = 0; pass < repeats; pass++)
                    for (int i = 0; i < N; i++)
                        store.fetch(keys[i].getNodeCHK(), false, false, null);
                return System.nanoTime() - t0;
            }
        }
    }
}
