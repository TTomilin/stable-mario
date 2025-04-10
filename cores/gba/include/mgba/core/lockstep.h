/* Copyright (c) 2013-2016 Jeffrey Pfau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#ifndef SIO_LOCKSTEP_H
#define SIO_LOCKSTEP_H

#include <mgba-util/common.h>

CXX_GUARD_START

enum mLockstepPhase {
	TRANSFER_IDLE = 0,
	TRANSFER_STARTING,
	TRANSFER_STARTED,
	TRANSFER_FINISHING,
	TRANSFER_FINISHED
};

struct mLockstep {
	int attached;
	enum mLockstepPhase transferActive;
	int32_t transferCycles;

	bool (*signal)(struct mLockstep*, unsigned mask);
	bool (*wait)(struct mLockstep*, unsigned mask);
	void (*addCycles)(struct mLockstep*, int id, int32_t cycles);
	int32_t (*useCycles)(struct mLockstep*, int id, int32_t cycles);
	void (*unload)(struct mLockstep*, int id);
	void* context;
#ifndef NDEBUG
	int transferId;
#endif
};

void mLockstepInit(struct mLockstep*);

CXX_GUARD_END

#endif
