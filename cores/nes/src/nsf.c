/* FCE Ultra - NES/Famicom Emulator
 *
 * Copyright notice for this file:
 *  Copyright (C) 2002 Xodnizel
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fceu-types.h"
#include "x6502.h"
#include "fceu.h"
#include "video.h"
#include "sound.h"
#include "nsf.h"
#include "general.h"
#include "fceu-memory.h"
#include "file.h"
#include "fds.h"
#include "cart.h"
#include "input.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint8 SongReload;
static int CurrentSong;

static DECLFW(NSF_write);
static DECLFR(NSF_read);

static int vismode = 1;

static uint8 NSFROM[0x30 + 6] =
{
/* 0x00 - NMI */
	0x8D, 0xF4, 0x3F,	/* Stop play routine NMIs. */
	0xA2, 0xFF, 0x9A,	/* Initialize the stack pointer. */
	0xAD, 0xF0, 0x3F,	/* See if we need to init. */
	0xF0, 0x09,			/* If 0, go to play routine playing. */

	0xAD, 0xF1, 0x3F,	/* Confirm and load A      */
	0xAE, 0xF3, 0x3F,	/* Load X with PAL/NTSC byte */

	0x20, 0x00, 0x00,	/* JSR to init routine     */

	0xA9, 0x00,
	0xAA,
	0xA8,
	0x20, 0x00, 0x00,	/* JSR to play routine  */
	0x8D, 0xF5, 0x3F,	/* Start play routine NMIs. */
	0x90, 0xFE,			/* Loopie time. */

/* 0x20 */
	0x8D, 0xF3, 0x3F,	/* Init init NMIs */
	0x18,
	0x90, 0xFE			/* Loopie time. */
};

static DECLFR(NSFROMRead) {
	return (NSFROM - 0x3800)[A];
}

static int doreset = 0;
static int NSFNMIFlags;
static uint8 *NSFDATA = 0;
static int NSFMaxBank;

static int NSFSize;
static uint8 BSon;
static uint16 PlayAddr;
static uint16 InitAddr;
static uint16 LoadAddr;

static NSF_HEADER NSFHeader;

void NSFMMC5_Close(void);
static uint8 *ExWRAM = 0;

void NSFGI(int h) {
	switch (h) {
	case GI_CLOSE:
		if (NSFDATA) {
			free(NSFDATA); NSFDATA = 0;
		}
		if (ExWRAM) {
			free(ExWRAM); ExWRAM = 0;
		}
		if (NSFHeader.SoundChip & 1) {
//   NSFVRC6_Init();
		} else if (NSFHeader.SoundChip & 2) {
//   NSFVRC7_Init();
		} else if (NSFHeader.SoundChip & 4) {
//   FDSSoundReset();
		} else if (NSFHeader.SoundChip & 8) {
			NSFMMC5_Close();
		} else if (NSFHeader.SoundChip & 0x10) {
//   NSFN106_Init();
		} else if (NSFHeader.SoundChip & 0x20) {
//   NSFAY_Init();
		}
		break;
	case GI_RESETM2:
	case GI_POWER: NSF_init(); break;
	}
}

// First 32KB is reserved for sound chip emulation in the iNES mapper code.

static INLINE void BANKSET(uint32 A, uint32 bank) {
	bank &= NSFMaxBank;
	if (NSFHeader.SoundChip & 4)
		memcpy(ExWRAM + (A - 0x6000), NSFDATA + (bank << 12), 4096);
	else
		setprg4(A, bank);
}

int NSFLoad(FCEUFILE *fp) {
	int x;

	FCEU_fseek(fp, 0, SEEK_SET);
	FCEU_fread(&NSFHeader, 1, 0x80, fp);
	if (memcmp(NSFHeader.ID, "NESM\x1a", 5))
		return 0;
	NSFHeader.SongName[31] = NSFHeader.Artist[31] = NSFHeader.Copyright[31] = 0;

	LoadAddr = NSFHeader.LoadAddressLow;
	LoadAddr |= NSFHeader.LoadAddressHigh << 8;

	if (LoadAddr < 0x6000) {
		FCEUD_PrintError("Invalid load address.");
		return(0);
	}
	InitAddr = NSFHeader.InitAddressLow;
	InitAddr |= NSFHeader.InitAddressHigh << 8;

	PlayAddr = NSFHeader.PlayAddressLow;
	PlayAddr |= NSFHeader.PlayAddressHigh << 8;

	NSFSize = FCEU_fgetsize(fp) - 0x80;

	NSFMaxBank = ((NSFSize + (LoadAddr & 0xfff) + 4095) / 4096);
	NSFMaxBank = uppow2(NSFMaxBank);

	if (!(NSFDATA = (uint8*)FCEU_malloc(NSFMaxBank * 4096)))
		return 0;

	FCEU_fseek(fp, 0x80, SEEK_SET);
	memset(NSFDATA, 0x00, NSFMaxBank * 4096);
	FCEU_fread(NSFDATA + (LoadAddr & 0xfff), 1, NSFSize, fp);

	NSFMaxBank--;

	BSon = 0;
	for (x = 0; x < 8; x++)
		BSon |= NSFHeader.BankSwitch[x];

	GameInfo->type = GIT_NSF;
	GameInfo->input[0] = GameInfo->input[1] = SI_GAMEPAD;
	GameInfo->cspecial = SIS_NSF;

	for (x = 0;; x++) {
		if (NSFROM[x] == 0x20) {
			NSFROM[x + 1] = InitAddr & 0xFF;
			NSFROM[x + 2] = InitAddr >> 8;
			NSFROM[x + 8] = PlayAddr & 0xFF;
			NSFROM[x + 9] = PlayAddr >> 8;
			break;
		}
	}

	if (NSFHeader.VideoSystem == 0)
		GameInfo->vidsys = GIV_NTSC;
	else if (NSFHeader.VideoSystem == 1)
		GameInfo->vidsys = GIV_PAL;

	GameInterface = NSFGI;

	FCEU_printf("NSF Loaded.  File information:\n\n");
	FCEU_printf(" Name:       %s\n Artist:     %s\n Copyright:  %s\n\n", NSFHeader.SongName, NSFHeader.Artist, NSFHeader.Copyright);
	if (NSFHeader.SoundChip) {
		static char *tab[6] = { "Konami VRCVI", "Konami VRCVII", "Nintendo FDS", "Nintendo MMC5", "Namco 106", "Sunsoft FME-07" };

		for (x = 0; x < 6; x++)
			if (NSFHeader.SoundChip & (1 << x)) {
				FCEU_printf(" Expansion hardware:  %s\n", tab[x]);
				NSFHeader.SoundChip = 1 << x;	/* Prevent confusing weirdness if more than one bit is set. */
				break;
			}
	}
	if (BSon)
		FCEU_printf(" Bank-switched.\n");
	FCEU_printf(" Load address:  $%04x\n Init address:  $%04x\n Play address:  $%04x\n", LoadAddr, InitAddr, PlayAddr);
	FCEU_printf(" %s\n", (NSFHeader.VideoSystem & 1) ? "PAL" : "NTSC");
	FCEU_printf(" Starting song:  %d / %d\n\n", NSFHeader.StartingSong, NSFHeader.TotalSongs);

	if (NSFHeader.SoundChip & 4)
		ExWRAM = FCEU_gmalloc(32768 + 8192);
	else
		ExWRAM = FCEU_gmalloc(8192);
	return 1;
}

static DECLFR(NSFVectorRead) {
	if (((NSFNMIFlags & 1) && SongReload) || (NSFNMIFlags & 2) || doreset) {
		if (A == 0xFFFA) return(0x00);
		else if (A == 0xFFFB) return(0x38);
		else if (A == 0xFFFC) return(0x20);
		else if (A == 0xFFFD) {
			doreset = 0; return(0x38);
		}
		return(X.DB);
	} else
		return(CartBR(A));
}

void NSFVRC6_Init(void);
void NSFVRC7_Init(void);
void NSFMMC5_Init(void);
void NSFN106_Init(void);
void NSFAY_Init(void);

void NSF_init(void) {
	doreset = 1;

	ResetCartMapping();
	if (NSFHeader.SoundChip & 4) {
		SetupCartPRGMapping(0, ExWRAM, 32768 + 8192, 1);
		setprg32(0x6000, 0);
		setprg8(0xE000, 4);
		memset(ExWRAM, 0x00, 32768 + 8192);
		SetWriteHandler(0x6000, 0xDFFF, CartBW);
		SetReadHandler(0x6000, 0xFFFF, CartBR);
	} else {
		memset(ExWRAM, 0x00, 8192);
		SetReadHandler(0x6000, 0x7FFF, CartBR);
		SetWriteHandler(0x6000, 0x7FFF, CartBW);
		SetupCartPRGMapping(0, NSFDATA, ((NSFMaxBank + 1) * 4096), 0);
		SetupCartPRGMapping(1, ExWRAM, 8192, 1);
		setprg8r(1, 0x6000, 0);
		SetReadHandler(0x8000, 0xFFFF, CartBR);
	}

	if (BSon) {
		int32 x;
		for (x = 0; x < 8; x++) {
			if (NSFHeader.SoundChip & 4 && x >= 6)
				BANKSET(0x6000 + (x - 6) * 4096, NSFHeader.BankSwitch[x]);
			BANKSET(0x8000 + x * 4096, NSFHeader.BankSwitch[x]);
		}
	} else {
		int32 x;
		for (x = (LoadAddr & 0xF000); x < 0x10000; x += 0x1000)
			BANKSET(x, ((x - (LoadAddr & 0x7000)) >> 12));
	}

	SetReadHandler(0xFFFA, 0xFFFD, NSFVectorRead);

	SetWriteHandler(0x2000, 0x3fff, 0);
	SetReadHandler(0x2000, 0x37ff, 0);
	SetReadHandler(0x3836, 0x3FFF, 0);
	SetReadHandler(0x3800, 0x3835, NSFROMRead);

	SetWriteHandler(0x5ff6, 0x5fff, NSF_write);

	SetWriteHandler(0x3ff0, 0x3fff, NSF_write);
	SetReadHandler(0x3ff0, 0x3fff, NSF_read);


	if (NSFHeader.SoundChip & 1) {
		NSFVRC6_Init();
	} else if (NSFHeader.SoundChip & 2) {
		NSFVRC7_Init();
	} else if (NSFHeader.SoundChip & 4) {
		FDSSoundReset();
	} else if (NSFHeader.SoundChip & 8) {
		NSFMMC5_Init();
	} else if (NSFHeader.SoundChip & 0x10) {
		NSFN106_Init();
	} else if (NSFHeader.SoundChip & 0x20) {
		NSFAY_Init();
	}
	CurrentSong = NSFHeader.StartingSong;
	SongReload = 0xFF;
	NSFNMIFlags = 0;
}

static DECLFW(NSF_write) {
	switch (A) {
	case 0x3FF3: NSFNMIFlags |= 1; break;
	case 0x3FF4: NSFNMIFlags &= ~2; break;
	case 0x3FF5: NSFNMIFlags |= 2; break;

	case 0x5FF6:
	case 0x5FF7: if (!(NSFHeader.SoundChip & 4)) return;
	case 0x5FF8:
	case 0x5FF9:
	case 0x5FFA:
	case 0x5FFB:
	case 0x5FFC:
	case 0x5FFD:
	case 0x5FFE:
	case 0x5FFF: if (!BSon) return;
		A &= 0xF;
		BANKSET((A * 4096), V);
		break;
	}
}

static DECLFR(NSF_read) {
	int x;

	switch (A) {
	case 0x3ff0: x = SongReload;
				#ifdef FCEUDEF_DEBUGGER
		if (!fceuindbg)
				#endif
		SongReload = 0;
		return x;
	case 0x3ff1:
			#ifdef FCEUDEF_DEBUGGER
		if (!fceuindbg)
			#endif
		{
			memset(RAM, 0x00, 0x800);

			BWrite[0x4015](0x4015, 0x0);
			for (x = 0; x < 0x14; x++)
				BWrite[0x4000 + x](0x4000 + x, 0);
			BWrite[0x4015](0x4015, 0xF);

			if (NSFHeader.SoundChip & 4) {
				BWrite[0x4017](0x4017, 0xC0);	/* FDS BIOS writes $C0 */
				BWrite[0x4089](0x4089, 0x80);
				BWrite[0x408A](0x408A, 0xE8);
			} else {
				memset(ExWRAM, 0x00, 8192);
				BWrite[0x4017](0x4017, 0xC0);
				BWrite[0x4017](0x4017, 0xC0);
				BWrite[0x4017](0x4017, 0x40);
			}

			if (BSon) {
				for (x = 0; x < 8; x++)
					BANKSET(0x8000 + x * 4096, NSFHeader.BankSwitch[x]);
			}
			return(CurrentSong - 1);
		}
	case 0x3FF3: return PAL;
	}
	return 0;
}

uint8 FCEU_GetJoyJoy(void);

static int special = 0;

void DrawNSF(uint8 *XBuf) {
	char snbuf[16];
	int x;

	if (vismode == 0) return;

	memset(XBuf, 0, 256 * 240);


	{
		int32 *Bufpl;
		int32 mul = 0;

		int l;
		l = GetSoundBuffer(&Bufpl);

		if (special == 0) {
			if (FSettings.SoundVolume)
				mul = 8192 * 240 / (16384 * FSettings.SoundVolume / 50);
			for (x = 0; x < 256; x++) {
				uint32 y;
				y = 142 + ((Bufpl[(x * l) >> 8] * mul) >> 14);
				if (y < 240)
					XBuf[x + y * 256] = 3;
			}
		} else if (special == 1) {
			if (FSettings.SoundVolume)
				mul = 8192 * 240 / (8192 * FSettings.SoundVolume / 50);
			for (x = 0; x < 256; x++) {
				double r;
				uint32 xp, yp;

				r = (Bufpl[(x * l) >> 8] * mul) >> 14;
				xp = 128 + r*cos(x*M_PI*2 / 256);
				yp = 120 + r*sin(x*M_PI*2 / 256);
				xp &= 255;
				yp %= 240;
				XBuf[xp + yp * 256] = 3;
			}
		} else if (special == 2) {
			static double theta = 0;
			if (FSettings.SoundVolume)
				mul = 8192 * 240 / (16384 * FSettings.SoundVolume / 50);
			for (x = 0; x < 128; x++) {
				double xc, yc;
				double r, t;
				uint32 m, n;

				xc = (double)128 - x;
				yc = 0 - ((double)(((Bufpl[(x * l) >> 8]) * mul) >> 14));
				t = M_PI + atan(yc / xc);
				r = sqrt(xc * xc + yc * yc);

				t += theta;
				m = 128 + r*cos(t);
				n = 120 + r*sin(t);

				if (m < 256 && n < 240)
					XBuf[m + n * 256] = 3;
			}
			for (x = 128; x < 256; x++) {
				double xc, yc;
				double r, t;
				uint32 m, n;

				xc = (double)x - 128;
				yc = (double)((Bufpl[(x * l) >> 8] * mul) >> 14);
				t = atan(yc / xc);
				r = sqrt(xc * xc + yc * yc);

				t += theta;
				m = 128 + r*cos(t);
				n = 120 + r*sin(t);

				if (m < 256 && n < 240)
					XBuf[m + n * 256] = 3;
			}
			theta += (double)M_PI / 256;
		}
	}

	DrawTextTrans(XBuf + 10 * 256 + 4 + (((31 - strlen((char*)NSFHeader.SongName)) << 2)), 256, NSFHeader.SongName, 6);
	DrawTextTrans(XBuf + 26 * 256 + 4 + (((31 - strlen((char*)NSFHeader.Artist)) << 2)), 256, NSFHeader.Artist, 6);
	DrawTextTrans(XBuf + 42 * 256 + 4 + (((31 - strlen((char*)NSFHeader.Copyright)) << 2)), 256, NSFHeader.Copyright, 6);

	DrawTextTrans(XBuf + 70 * 256 + 4 + (((31 - strlen("Song:")) << 2)), 256, (uint8*)"Song:", 6);
	sprintf(snbuf, "<%d/%d>", CurrentSong, NSFHeader.TotalSongs);
	DrawTextTrans(XBuf + 82 * 256 + 4 + (((31 - strlen(snbuf)) << 2)), 256, (uint8*)snbuf, 6);

	{
		static uint8 last = 0;
		uint8 tmp;
		tmp = FCEU_GetJoyJoy();
		if ((tmp & JOY_RIGHT) && !(last & JOY_RIGHT)) {
			if (CurrentSong < NSFHeader.TotalSongs) {
				CurrentSong++;
				SongReload = 0xFF;
			}
		} else if ((tmp & JOY_LEFT) && !(last & JOY_LEFT)) {
			if (CurrentSong > 1) {
				CurrentSong--;
				SongReload = 0xFF;
			}
		} else if ((tmp & JOY_UP) && !(last & JOY_UP)) {
			CurrentSong += 10;
			if (CurrentSong > NSFHeader.TotalSongs) CurrentSong = NSFHeader.TotalSongs;
			SongReload = 0xFF;
		} else if ((tmp & JOY_DOWN) && !(last & JOY_DOWN)) {
			CurrentSong -= 10;
			if (CurrentSong < 1) CurrentSong = 1;
			SongReload = 0xFF;
		} else if ((tmp & JOY_START) && !(last & JOY_START))
			SongReload = 0xFF;
		else if ((tmp & JOY_A) && !(last & JOY_A)) {
			special = (special + 1) % 3;
		}
		last = tmp;
	}
}

void DoNSFFrame(void) {
	if (((NSFNMIFlags & 1) && SongReload) || (NSFNMIFlags & 2))
		TriggerNMI();
}

void FCEUI_NSFSetVis(int mode) {
	vismode = mode;
}

int FCEUI_NSFChange(int amount) {
	CurrentSong += amount;
	if (CurrentSong < 1) CurrentSong = 1;
	else if (CurrentSong > NSFHeader.TotalSongs) CurrentSong = NSFHeader.TotalSongs;
	SongReload = 0xFF;

	return(CurrentSong);
}

/* Returns total songs */
int FCEUI_NSFGetInfo(uint8 *name, uint8 *artist, uint8 *copyright, int maxlen) {
	strncpy((char*)name, (const char*)NSFHeader.SongName, (size_t)maxlen);
	strncpy((char*)artist, (const char*)NSFHeader.Artist, (size_t)maxlen);
	strncpy((char*)copyright, (const char*)NSFHeader.Copyright, (size_t)maxlen);
	return(NSFHeader.TotalSongs);
}
