/* Copyright (c) 2013-2016 Jeffrey Pfau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#ifndef GB_CHEATS_H
#define GB_CHEATS_H

#include <mgba-util/common.h>

CXX_GUARD_START

#include <mgba/core/cheats.h>
#include <mgba-util/vector.h>

enum GBCheatType {
	GB_CHEAT_AUTODETECT,
	GB_CHEAT_GAMESHARK,
	GB_CHEAT_GAME_GENIE,
	GB_CHEAT_VBA
};

struct GBCheatPatch {
	uint16_t address;
	int8_t newValue;
	int8_t oldValue;
	int segment;
	bool applied;
	bool checkByte;
};

DECLARE_VECTOR(GBCheatPatchList, struct GBCheatPatch);

struct GBCheatSet {
	struct mCheatSet d;
	struct GBCheatPatchList romPatches;
};

struct mCheatDevice* GBCheatDeviceCreate(void);

CXX_GUARD_END

#endif
