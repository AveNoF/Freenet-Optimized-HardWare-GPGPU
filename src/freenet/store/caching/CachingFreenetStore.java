package freenet.store.caching;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import freenet.keys.KeyVerifyException;
import freenet.node.SemiOrderedShutdownHook;
import freenet.store.BlockMetadata;
import freenet.store.FreenetStore;
import freenet.store.KeyCollisionException;
import freenet.store.ProxyFreenetStore;
import freenet.store.StorableBlock;
import freenet.store.StoreCallback;
import freenet.support.ByteArrayWrapper;
import freenet.support.LRUMap;
import freenet.support.Logger;
import freenet.support.Ticker;
import freenet.support.io.NativeThread;

/**
 * CachingFreenetStore
 * 
 * @author Simon Vocella <voxsim@gmail.com>
 * 
 */
public class CachingFreenetStore<T extends StorableBlock> extends ProxyFreenetStore<T> {
    private static volatile boolean logMINOR;
 
	private boolean shuttingDown; /* If this flag is true, we don't accept puts anymore */
	/***
	 * True if close() has been called
	 */
	private AtomicBoolean closeCalled = new AtomicBoolean(false);

	/** Write-back buffer: blocks that have been put() but not yet flushed to the underlying store. */
	private final LRUMap<ByteArrayWrapper, Block<T>> blocksByRoutingKey;
	/** Read-through cache: clean blocks that were fetched from the underlying store and are kept in
	 * memory to avoid repeated random disk I/O. These are NEVER written back (they are already on
	 * disk); they are only evicted (LRU). Entries here have {@link Block#block} set and
	 * {@link Block#data}/{@link Block#header} null. */
	private final LRUMap<ByteArrayWrapper, Block<T>> readCache;
	/** Bytes currently held by this instance's read-through cache (== readCache.size()*sizeBlock). */
	private long readCacheBytes;
	private final StoreCallback<T> callback;
	private final boolean collisionPossible;
	private final ReadWriteLock configLock = new ReentrantReadWriteLock();
	private final CachingFreenetStoreTracker tracker;
	private final int sizeBlock;
	
    static { Logger.registerClass(CachingFreenetStore.class); }
    
	private final static class Block<T> {
		T block;
		byte[] data;
		byte[] header;
		boolean overwrite;
		boolean isOldBlock;
	}

	public CachingFreenetStore(StoreCallback<T> callback, FreenetStore<T> backDatastore, CachingFreenetStoreTracker tracker) {
		super(backDatastore);
		this.callback = callback;
		SemiOrderedShutdownHook shutdownHook = SemiOrderedShutdownHook.get();
		this.blocksByRoutingKey = LRUMap.createSafeMap(ByteArrayWrapper.FAST_COMPARATOR);
		this.readCache = LRUMap.createSafeMap(ByteArrayWrapper.FAST_COMPARATOR);
		this.collisionPossible = callback.collisionPossible();
		this.shuttingDown = false;
		this.tracker = tracker;
		this.sizeBlock = callback.getTotalBlockSize();
		
		callback.setStore(this);
		shutdownHook.addEarlyJob(new NativeThread("Close CachingFreenetStore", NativeThread.HIGH_PRIORITY, true) {
			@Override
			public void realRun() {
				innerClose(); // SaltedHashFS has its own shutdown job.
			}
		});
	}

	@Override
	public T fetch(byte[] routingKey, byte[] fullKey,
			boolean dontPromote, boolean canReadClientCache,
			boolean canReadSlashdotCache, boolean ignoreOldBlocks, BlockMetadata meta) 
			throws IOException {
		ByteArrayWrapper key = new ByteArrayWrapper(routingKey);
		
		Block<T> writeBack = null;
		Block<T> cached = null;
		
		configLock.readLock().lock();
		try {
			writeBack = blocksByRoutingKey.get(key);
			if(writeBack == null)
				cached = readCache.get(key);
		} finally {
			configLock.readLock().unlock();
		}
		
		// Pending write-back block: reconstruct exactly as before (preserves old behaviour).
		if(writeBack != null) {
			try {
				return this.callback.construct(writeBack.data, writeBack.header, routingKey, writeBack.block.getFullKey(), canReadClientCache, canReadSlashdotCache, meta, null);
			} catch (KeyVerifyException e) {
				Logger.error(this, "Error in fetching for CachingFreenetStore: "+e, e);
			}
		}
		
		// Read-through cache hit. For stores where the same routing key can map to different blocks
		// (SSK), only serve from cache if the full key matches, otherwise we might return the wrong
		// block.
		if(cached != null && cached.block != null) {
			boolean safe = !collisionPossible ||
				(fullKey != null && Arrays.equals(fullKey, cached.block.getFullKey()));
			if(safe) {
				if(meta != null && cached.isOldBlock) meta.setOldBlock();
				if(!dontPromote) {
					configLock.writeLock().lock();
					try {
						// Re-check: the entry may have been evicted in the meantime.
						if(readCache.get(key) == cached)
							readCache.push(key, cached);
					} finally {
						configLock.writeLock().unlock();
					}
				}
				return cached.block;
			}
		}
		
		// Miss: hit the underlying store, then cache the result for next time.
		T result = backDatastore.fetch(routingKey, fullKey, dontPromote, canReadClientCache, canReadSlashdotCache, ignoreOldBlocks, meta);
		if(result != null && !dontPromote)
			putReadCache(key, result, meta != null && meta.isOldBlock());
		return result;
	}

	/** Add a clean (already-on-disk) block to the read-through cache, evicting least-recently-used
	 * entries if we are over the shared read-cache budget. */
	private void putReadCache(ByteArrayWrapper key, T result, boolean isOldBlock) {
		if(tracker.getReadCacheLimit() <= 0) return; // Read-through caching disabled.
		Block<T> entry = new Block<T>();
		entry.block = result;
		entry.isOldBlock = isOldBlock;
		// data/header intentionally left null: this marks it as a read-through (clean) entry.
		configLock.writeLock().lock();
		try {
			if(shuttingDown) return;
			// Don't shadow a pending write-back block.
			if(blocksByRoutingKey.get(key) != null) return;
			boolean wasPresent = readCache.get(key) != null;
			readCache.push(key, entry);
			if(!wasPresent) {
				readCacheBytes += sizeBlock;
				tracker.addReadCache(sizeBlock);
			}
			// Evict our own LRU entries while the global read cache is over budget.
			while(tracker.getReadCacheSize() > tracker.getReadCacheLimit()) {
				if(readCache.popValue() == null) break;
				readCacheBytes -= sizeBlock;
				tracker.removeReadCache(sizeBlock);
			}
		} finally {
			configLock.writeLock().unlock();
		}
	}

	@Override
	public boolean probablyInStore(byte[] routingKey) {
		ByteArrayWrapper key = new ByteArrayWrapper(routingKey);
		Block<T> block = null;
		
		configLock.readLock().lock();
		try {
			block = blocksByRoutingKey.get(key);
		} finally {
			configLock.readLock().unlock();
		}
		
		return block != null || backDatastore.probablyInStore(routingKey);
	}
	
	@Override
	public void put(T block, byte[] data, byte[] header,
			boolean overwrite, boolean isOldBlock) throws IOException, KeyCollisionException {
		byte[] routingKey = block.getRoutingKey();
		final ByteArrayWrapper key = new ByteArrayWrapper(routingKey);
		
		Block<T> storeBlock = new Block<T>();
		storeBlock.block = block;
		storeBlock.data = data;
		storeBlock.header = header;
		storeBlock.overwrite = overwrite;
		storeBlock.isOldBlock = isOldBlock;
		
		boolean cacheIt = true;
		
		//Case cache it
		configLock.writeLock().lock();
		
		try {
			// A newer version of this block is being stored; drop any stale read-through copy.
			if(readCache.removeKey(key)) {
				readCacheBytes -= sizeBlock;
				tracker.removeReadCache(sizeBlock);
			}
			if(!shuttingDown) {
				Block<T> previousBlock = blocksByRoutingKey.get(key);
			
				if(!collisionPossible || overwrite) {
					if(previousBlock == null) {
						cacheIt = tracker.add(sizeBlock);
					}
					
					if(cacheIt) {
						blocksByRoutingKey.push(key, storeBlock);
					}
				} else {
					//Case cache it but is it in blocksByRoutingKey? If so, throw a KCE
					if(previousBlock != null) {
						if(block.equals(previousBlock.block))
							return;
						throw new KeyCollisionException();
					}
					
					//Is probablyInStore()? If so, remove it from blocksByRoutingKey, and set a flag so we don't call put()
					if(backDatastore.probablyInStore(routingKey)) {
						cacheIt = false;
					} else {
						cacheIt = tracker.add(sizeBlock);
						
						if(cacheIt) {
							blocksByRoutingKey.push(key, storeBlock);
						}
					}
				}
			} else {
				cacheIt = false;
			}
		} finally {
			configLock.writeLock().unlock();
		}
		
		//Case don't cache it
		if(!cacheIt) {
			backDatastore.put(block, data, header, overwrite, isOldBlock);
			return;
		}
	}
	
	/** Try to write one block to disk.
	 * @return The number of bytes written to disk if we successfully wrote a block, 0 if we wrote 
	 * a block but can't remove it because it changed while we were writing it, and -1 if there 
	 * were no blocks to write because the cache is empty.
	 */
	long pushLeastRecentlyBlock() {
		Block<T> block = null;
		ByteArrayWrapper key = null;
		
		configLock.writeLock().lock();
		try {
			block = blocksByRoutingKey.peekValue();
			if(block == null) return -1;
			key = blocksByRoutingKey.peekKey();
		} finally {
			configLock.writeLock().unlock();
		}
			
		try {
			backDatastore.put(block.block, block.data, block.header, block.overwrite, block.isOldBlock);
		} catch (IOException e) {
			Logger.error(this, "Error in pushAll for CachingFreenetStore: "+e, e);
		} catch (KeyCollisionException e) {
			if(logMINOR) Logger.minor(this, "KeyCollisionException in pushAll for CachingFreenetStore: "+e, e);
		}
		
		configLock.writeLock().lock();
		try {
			Block<T> currentVersionOfBlock = blocksByRoutingKey.get(key);
			
			/** it might have changed if there was a put() with overwrite=true. 
			 *  If it has changed, return 0 , i.e. don't remove it*/
			if(currentVersionOfBlock != null && currentVersionOfBlock.block.equals(block.block)) {
				if(blocksByRoutingKey.removeKey(key))
					return sizeBlock;
			}
		} finally {
			configLock.writeLock().unlock();
		}
		return 0;
	}

	@Override
	public boolean start(Ticker ticker, boolean longStart) throws IOException {
		tracker.registerCachingFS(this);
		return this.backDatastore.start(ticker, longStart);
	}

	@Override
	public void close() {
		if( closeCalled.compareAndSet(false, true)) {
			innerClose();
			backDatastore.close();
		}
	}

	/** Close this store but not the underlying store. */
	private void innerClose() {
		configLock.writeLock().lock();
		try {
			shuttingDown = true;
			// Drop the read-through cache and return its budget to the tracker.
			while(readCache.popValue() != null) {
				readCacheBytes -= sizeBlock;
				tracker.removeReadCache(sizeBlock);
			}
			tracker.unregisterCachingFS(this);
		} finally {
			configLock.writeLock().unlock();
		}
	}
	
	/** Only for unit tests */
	boolean isEmpty() {
		boolean isEmpty;
		configLock.readLock().lock();
		try {
			isEmpty = this.blocksByRoutingKey.isEmpty();
		} finally {
			configLock.readLock().unlock();
		}
		return isEmpty;
	}
}
