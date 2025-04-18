/* FCE Ultra - NES/Famicom Emulator
 *
 * Copyright notice for this file:
 *	Copyright (C) 2015 Cluster
 *	http://clusterrr.com
 *	clusterrr@clusterrr.com
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA	02110-1301	USA
 */

/*
	MMC3-based multicart mapper with CHR RAM, CHR ROM and PRG RAM

	$6000-7FFF:	A~[011xxxxx xxMRSBBB]	Multicart reg
		This register can only be written to if PRG-RAM is enabled and writable (see $A001)
		and BBB = 000 (power on state)

	BBB = CHR+PRG block select bits (A19, A18, A17 for both PRG and CHR)
	S = PRG block size (0=256k	 1=128k)
	R = CHR mode (0=CHR ROM	 1=CHR RAM)
	M = CHR block size (0=256k	 1=128k)
		ignored when S is 0 for some reason

 Example Game:
 --------------------------
 6 in 1 multicart (SMB3, TMNT2, Contra, Ninja Cat, Ninja Crusaders, Rainbow Islands 2)
*/

#include "mapinc.h"
#include "mmc3.h"

static uint8 *CHRRAM = NULL;
static uint32 CHRRAMSize;

static void BMC1024CA1PW(uint32 A, uint8 V) {
	if ((EXPREGS[0]>>3)&1)
		setprg8(A, (V&0x1F) | ((EXPREGS[0] & 7) << 4));
	else
		setprg8(A, (V&0x0F) | ((EXPREGS[0] & 7) << 4));
}

static void BMC1024CA1CW(uint32 A, uint8 V) {
	if ((EXPREGS[0]>>4)&1)
	 	setchr1r(0x10, A, V);
	else if (((EXPREGS[0]>>5)&1) && ((EXPREGS[0]>>3)&1))
		setchr1(A, V | ((EXPREGS[0] & 7) << 7));
	else
	 	setchr1(A, (V&0x7F) | ((EXPREGS[0] & 7) << 7));
}

static DECLFW(BMC1024CA1Write) {
	if (((A001B & 0xC0) == 0x80) && !(EXPREGS[0] & 7))
	{
		EXPREGS[0] = A & 0x3F;
		FixMMC3PRG(MMC3_cmd);
		FixMMC3CHR(MMC3_cmd);
	} else {
		CartBW(A, V);
	}
}

static void BMC1024CA1Reset(void) {
	EXPREGS[0] = 0;
	MMC3RegReset();
}

static void BMC1024CA1Power(void) {
	EXPREGS[0] = 0;
	GenMMC3Power();
	SetWriteHandler(0x6000, 0x7FFF, BMC1024CA1Write);
}

static void BMC1024CA1Close(void) {
	if (CHRRAM)
		FCEU_gfree(CHRRAM);
	CHRRAM = NULL;
}

void BMC1024CA1_Init(CartInfo *info) {
	GenMMC3_Init(info, 256, 256, 8, 0);
	CHRRAMSize = 8192;
	CHRRAM = (uint8*)FCEU_gmalloc(CHRRAMSize);
	SetupCartCHRMapping(0x10, CHRRAM, CHRRAMSize, 1);
	AddExState(CHRRAM, CHRRAMSize, 0, "CHRR");
	pwrap = BMC1024CA1PW;
	cwrap = BMC1024CA1CW;
	info->Power = BMC1024CA1Power;
	info->Reset = BMC1024CA1Reset;
	info->Close = BMC1024CA1Close;
	AddExState(EXPREGS, 1, 0, "EXPR");
}
