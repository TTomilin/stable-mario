/***********************************************************************************
  Snes9x - Portable Super Nintendo Entertainment System (TM) emulator.

  (c) Copyright 1996 - 2002  Gary Henderson (gary.henderson@ntlworld.com),
                             Jerremy Koot (jkoot@snes9x.com)

  (c) Copyright 2002 - 2004  Matthew Kendora

  (c) Copyright 2002 - 2005  Peter Bortas (peter@bortas.org)

  (c) Copyright 2004 - 2005  Joel Yliluoma (http://iki.fi/bisqwit/)

  (c) Copyright 2001 - 2006  John Weidman (jweidman@slip.net)

  (c) Copyright 2002 - 2006  funkyass (funkyass@spam.shaw.ca),
                             Kris Bleakley (codeviolation@hotmail.com)

  (c) Copyright 2002 - 2010  Brad Jorsch (anomie@users.sourceforge.net),
                             Nach (n-a-c-h@users.sourceforge.net),

  (c) Copyright 2002 - 2011  zones (kasumitokoduck@yahoo.com)

  (c) Copyright 2006 - 2007  nitsuja

  (c) Copyright 2009 - 2011  BearOso,
                             OV2

  (c) Copyright 2011 - 2016  Hans-Kristian Arntzen,
                             Daniel De Matteis
                             (Under no circumstances will commercial rights be given)


  BS-X C emulator code
  (c) Copyright 2005 - 2006  Dreamer Nom,
                             zones

  C4 x86 assembler and some C emulation code
  (c) Copyright 2000 - 2003  _Demo_ (_demo_@zsnes.com),
                             Nach,
                             zsKnight (zsknight@zsnes.com)

  C4 C++ code
  (c) Copyright 2003 - 2006  Brad Jorsch,
                             Nach

  DSP-1 emulator code
  (c) Copyright 1998 - 2006  _Demo_,
                             Andreas Naive (andreasnaive@gmail.com),
                             Gary Henderson,
                             Ivar (ivar@snes9x.com),
                             John Weidman,
                             Kris Bleakley,
                             Matthew Kendora,
                             Nach,
                             neviksti (neviksti@hotmail.com)

  DSP-2 emulator code
  (c) Copyright 2003         John Weidman,
                             Kris Bleakley,
                             Lord Nightmare (lord_nightmare@users.sourceforge.net),
                             Matthew Kendora,
                             neviksti

  DSP-3 emulator code
  (c) Copyright 2003 - 2006  John Weidman,
                             Kris Bleakley,
                             Lancer,
                             z80 gaiden

  DSP-4 emulator code
  (c) Copyright 2004 - 2006  Dreamer Nom,
                             John Weidman,
                             Kris Bleakley,
                             Nach,
                             z80 gaiden

  OBC1 emulator code
  (c) Copyright 2001 - 2004  zsKnight,
                             pagefault (pagefault@zsnes.com),
                             Kris Bleakley
                             Ported from x86 assembler to C by sanmaiwashi

  SPC7110 and RTC C++ emulator code used in 1.39-1.51
  (c) Copyright 2002         Matthew Kendora with research by
                             zsKnight,
                             John Weidman,
                             Dark Force

  SPC7110 and RTC C++ emulator code used in 1.52+
  (c) Copyright 2009         byuu,
                             neviksti

  S-DD1 C emulator code
  (c) Copyright 2003         Brad Jorsch with research by
                             Andreas Naive,
                             John Weidman

  S-RTC C emulator code
  (c) Copyright 2001 - 2006  byuu,
                             John Weidman

  ST010 C++ emulator code
  (c) Copyright 2003         Feather,
                             John Weidman,
                             Kris Bleakley,
                             Matthew Kendora

  Super FX x86 assembler emulator code
  (c) Copyright 1998 - 2003  _Demo_,
                             pagefault,
                             zsKnight

  Super FX C emulator code
  (c) Copyright 1997 - 1999  Ivar,
                             Gary Henderson,
                             John Weidman

  Sound emulator code used in 1.5-1.51
  (c) Copyright 1998 - 2003  Brad Martin
  (c) Copyright 1998 - 2006  Charles Bilyue'

  Sound emulator code used in 1.52+
  (c) Copyright 2004 - 2007  Shay Green (gblargg@gmail.com)

  SH assembler code partly based on x86 assembler code
  (c) Copyright 2002 - 2004  Marcus Comstedt (marcus@mc.pp.se)

  2xSaI filter
  (c) Copyright 1999 - 2001  Derek Liauw Kie Fa

  HQ2x, HQ3x, HQ4x filters
  (c) Copyright 2003         Maxim Stepin (maxim@hiend3d.com)

  NTSC filter
  (c) Copyright 2006 - 2007  Shay Green

  GTK+ GUI code
  (c) Copyright 2004 - 2011  BearOso

  Win32 GUI code
  (c) Copyright 2003 - 2006  blip,
                             funkyass,
                             Matthew Kendora,
                             Nach,
                             nitsuja
  (c) Copyright 2009 - 2011  OV2

  Mac OS GUI code
  (c) Copyright 1998 - 2001  John Stiles
  (c) Copyright 2001 - 2011  zones

  Libretro port
  (c) Copyright 2011 - 2016  Hans-Kristian Arntzen,
                             Daniel De Matteis
                             (Under no circumstances will commercial rights be given)


  Specific ports contains the works of other authors. See headers in
  individual files.


  Snes9x homepage: http://www.snes9x.com/

  Permission to use, copy, modify and/or distribute Snes9x in both binary
  and source form, for non-commercial purposes, is hereby granted without
  fee, providing that this license information and copyright notice appear
  with all copies and any derived work.

  This software is provided 'as-is', without any express or implied
  warranty. In no event shall the authors be held liable for any damages
  arising from the use of this software or it's derivatives.

  Snes9x is freeware for PERSONAL USE only. Commercial users should
  seek permission of the copyright holders first. Commercial use includes,
  but is not limited to, charging money for Snes9x or software derived from
  Snes9x, including Snes9x or derivatives in commercial game bundles, and/or
  using Snes9x as a promotion for your commercial product.

  The copyright holders request that bug fixes and improvements to the code
  should be forwarded to them so everyone can benefit from the modifications
  in future versions.

  Super NES and Super Nintendo Entertainment System are trademarks of
  Nintendo Co., Limited and its subsidiary companies.
 ***********************************************************************************/


#include "snes9x.h"
#include "memmap.h"
#include "seta.h"

static uint8	board[9][9];	// shougi playboard
static int		line = 0;		// line counter


uint8 S9xGetST011 (uint32 Address)
{
	uint8	t;
	uint16	address = (uint16) Address & 0xFFFF;

	line++;

	// status check
	if (address == 0x01)
		t = 0xFF;
	else
		t = Memory.SRAM[address]; // read directly from s-ram

#ifdef DEBUGGER
	if (address < 0x150)
		printf("ST011 R: %06X %02X\n", Address, t);
#endif

	return (t);
}

void S9xSetST011 (uint32 Address, uint8 Byte)
{
	static bool	reset   = false;
	uint16		address = (uint16) Address & 0xFFFF;

	line++;

	if (!reset)
	{
		// bootup values
		ST011.waiting4command = true;
		reset = true;
	}

#ifdef DEBUGGER
	if (address < 0x150)
		printf("ST011 W: %06X %02X\n", Address, Byte);
#endif

	Memory.SRAM[address] = Byte;

	// op commands/data goes through this address
	if (address == 0x00)
	{
		// check for new commands
		if (ST011.waiting4command)
		{
			ST011.waiting4command = false;
			ST011.command         = Byte;
			ST011.in_index        = 0;
			ST011.out_index       = 0;

			switch (ST011.command)
			{
				case 0x01: ST011.in_count = 12 * 10 + 8; break;
				case 0x02: ST011.in_count = 4;           break;
				case 0x04: ST011.in_count = 0;           break;
				case 0x05: ST011.in_count = 0;           break;
				case 0x06: ST011.in_count = 0;           break;
				case 0x07: ST011.in_count = 0;           break;
				case 0x0E: ST011.in_count = 0;           break;
				default:   ST011.waiting4command = true; break;
			}
		}
		else
		{
			ST011.parameters[ST011.in_index] = Byte;
			ST011.in_index++;
		}
	}

	if (ST011.in_count == ST011.in_index)
	{
		// actually execute the command
		ST011.waiting4command = true;
		ST011.out_index       = 0;

		switch (ST011.command)
		{
			// unknown: download playboard
			case 0x01:
				// 9x9 board data: top to bottom, left to right
				// Values represent piece types and ownership
				for (int lcv = 0; lcv < 9; lcv++)
					memcpy(board[lcv], ST011.parameters + lcv * 10, 9 * 1);
				break;

			// unknown
			case 0x02:
				break;

			// unknown
			case 0x04:
				// outputs
				Memory.SRAM[0x12C] = 0x00;
				//Memory.SRAM[0x12D] = 0x00;
				Memory.SRAM[0x12E] = 0x00;
				break;

			// unknown
			case 0x05:
				// outputs
				Memory.SRAM[0x12C] = 0x00;
				//Memory.SRAM[0x12D] = 0x00;
				Memory.SRAM[0x12E] = 0x00;
				break;

			// unknown
			case 0x06:
				break;

			case 0x07:
				break;

			// unknown
			case 0x0E:
				// outputs
				Memory.SRAM[0x12C] = 0x00;
				Memory.SRAM[0x12D] = 0x00;
				break;
		}
	}
}
