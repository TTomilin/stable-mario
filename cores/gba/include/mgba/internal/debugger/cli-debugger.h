/* Copyright (c) 2013-2016 Jeffrey Pfau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#ifndef CLI_DEBUGGER_H
#define CLI_DEBUGGER_H

#include <mgba-util/common.h>

CXX_GUARD_START

#include <mgba/debugger/debugger.h>

extern const char* ERROR_MISSING_ARGS;
extern const char* ERROR_OVERFLOW;
extern const char* ERROR_INVALID_ARGS;

struct CLIDebugger;

struct CLIDebugVector {
	struct CLIDebugVector* next;
	enum CLIDVType {
		CLIDV_ERROR_TYPE,
		CLIDV_INT_TYPE,
		CLIDV_CHAR_TYPE,
	} type;
	char* charValue;
	int32_t intValue;
	int segmentValue;
};

typedef void (*CLIDebuggerCommand)(struct CLIDebugger*, struct CLIDebugVector*);

struct CLIDebuggerCommandSummary {
	const char* name;
	CLIDebuggerCommand command;
	const char* format;
	const char* summary;
};

struct CLIDebuggerSystem {
	struct CLIDebugger* p;

	void (*init)(struct CLIDebuggerSystem*);
	void (*deinit)(struct CLIDebuggerSystem*);
	bool (*custom)(struct CLIDebuggerSystem*);

	void (*disassemble)(struct CLIDebuggerSystem*, struct CLIDebugVector* dv);
	void (*printStatus)(struct CLIDebuggerSystem*);

	struct CLIDebuggerCommandSummary* commands;
	const char* name;
	struct CLIDebuggerCommandSummary* platformCommands;
	const char* platformName;
};

struct CLIDebuggerBackend {
	struct CLIDebugger* p;

	void (*init)(struct CLIDebuggerBackend*);
	void (*deinit)(struct CLIDebuggerBackend*);

	ATTRIBUTE_FORMAT(printf, 2, 3)
	void (*printf)(struct CLIDebuggerBackend*, const char* fmt, ...);
	const char* (*readline)(struct CLIDebuggerBackend*, size_t* len);
	void (*lineAppend)(struct CLIDebuggerBackend*, const char* line);
	const char* (*historyLast)(struct CLIDebuggerBackend*, size_t* len);
	void (*historyAppend)(struct CLIDebuggerBackend*, const char* line);
};

struct CLIDebugger {
	struct mDebugger d;

	struct CLIDebuggerSystem* system;
	struct CLIDebuggerBackend* backend;
};

void CLIDebuggerCreate(struct CLIDebugger*);
void CLIDebuggerAttachSystem(struct CLIDebugger*, struct CLIDebuggerSystem*);
void CLIDebuggerAttachBackend(struct CLIDebugger*, struct CLIDebuggerBackend*);

bool CLIDebuggerTabComplete(struct CLIDebugger*, const char* token, bool initial, size_t len);

CXX_GUARD_END

#endif
