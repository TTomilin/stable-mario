/* Copyright (c) 2013-2017 Jeffrey Pfau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#include <mgba/core/cache-set.h>

DEFINE_VECTOR(mMapCacheSet, struct mMapCache);
DEFINE_VECTOR(mTileCacheSet, struct mTileCache);

void mCacheSetInit(struct mCacheSet* cache, size_t nMaps, size_t nTiles) {
	mMapCacheSetInit(&cache->maps, nMaps);
	mMapCacheSetResize(&cache->maps, nMaps);
	mTileCacheSetInit(&cache->tiles, nTiles);
	mTileCacheSetResize(&cache->tiles, nTiles);

	size_t i;
	for (i = 0; i < nMaps; ++i) {
		mMapCacheInit(mMapCacheSetGetPointer(&cache->maps, i));
	}
	for (i = 0; i < nTiles; ++i) {
		mTileCacheInit(mTileCacheSetGetPointer(&cache->tiles, i));
	}
}

void mCacheSetDeinit(struct mCacheSet* cache) {
	size_t i;
	for (i = 0; i < mMapCacheSetSize(&cache->maps); ++i) {
		mMapCacheDeinit(mMapCacheSetGetPointer(&cache->maps, i));
	}
	for (i = 0; i < mTileCacheSetSize(&cache->tiles); ++i) {
		mTileCacheDeinit(mTileCacheSetGetPointer(&cache->tiles, i));
	}
}

void mCacheSetAssignVRAM(struct mCacheSet* cache, void* vram) {
	size_t i;
	for (i = 0; i < mMapCacheSetSize(&cache->maps); ++i) {
		mMapCacheSetGetPointer(&cache->maps, i)->vram = vram;
	}
	for (i = 0; i < mTileCacheSetSize(&cache->tiles); ++i) {
		struct mTileCache* tileCache = mTileCacheSetGetPointer(&cache->tiles, i);
		tileCache->vram = (void*) ((uintptr_t) vram + tileCache->tileBase);
	}
}

void mCacheSetWriteVRAM(struct mCacheSet* cache, uint32_t address) {
	size_t i;
	for (i = 0; i < mMapCacheSetSize(&cache->maps); ++i) {
		mMapCacheWriteVRAM(mMapCacheSetGetPointer(&cache->maps, i), address);
	}
	for (i = 0; i < mTileCacheSetSize(&cache->tiles); ++i) {
		mTileCacheWriteVRAM(mTileCacheSetGetPointer(&cache->tiles, i), address);
	}
}

void mCacheSetWritePalette(struct mCacheSet* cache, uint32_t entry, color_t color) {
	size_t i;
	for (i = 0; i < mTileCacheSetSize(&cache->tiles); ++i) {
		mTileCacheWritePalette(mTileCacheSetGetPointer(&cache->tiles, i), entry, color);
	}
}
