/* Copyright (c) 2013-2017 Jeffrey Pfau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#ifndef PYTHON_ENGINE_H
#define PYTHON_ENGINE_H

#include <mgba-util/common.h>

CXX_GUARD_START

struct mScriptBridge;
struct mPythonScriptEngine;
struct mPythonScriptEngine* mPythonCreateScriptEngine(void);
void mPythonSetup(struct mScriptBridge* sb);

CXX_GUARD_END

#endif
