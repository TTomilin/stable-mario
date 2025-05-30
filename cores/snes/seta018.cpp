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

static int	line;	// line counter


uint8 S9xGetST018 (uint32 Address)
{
	uint8	t       = 0;
	uint16	address = (uint16) Address & 0xFFFF;

	line++;

	// these roles may be flipped
	// op output
	if (address == 0x3804)
	{
		if (ST018.out_count)
		{
			t = (uint8) ST018.output[ST018.out_index];
			ST018.out_index++;
			if (ST018.out_count == ST018.out_index)
				ST018.out_count = 0;
		}
		else
			t = 0x81;
	}
	// status register
	else
	if (address == 0x3800)
		t = ST018.status;

#ifdef DEBUGGER
	printf("ST018 R: %06X %02X\n", Address, t);
#endif

	return (t);
}

void S9xSetST018 (uint8 Byte, uint32 Address)
{
	static bool	reset   = false;
	uint16		address = (uint16) Address & 0xFFFF;

#ifdef DEBUGGER
	printf("ST018 W: %06X %02X\n", Address, Byte);
#endif

	line++;

	if (!reset)
	{
		// bootup values
		ST018.waiting4command = true;
		ST018.part_command    = 0;
		reset = true;
	}

	Memory.SRAM[address] = Byte;

	// default status for now
	ST018.status = 0x00;

	// op data goes through this address
	if (address == 0x3804)
	{
		// check for new commands: 3 bytes length
		if (ST018.waiting4command && ST018.part_command == 2)
		{
			ST018.waiting4command = false;
			ST018.in_index        = 0;
			ST018.out_index       = 0;
			ST018.part_command    = 0; // 3-byte commands
			ST018.pass            = 0; // data streams into the chip
			ST018.command <<= 8;
			ST018.command |= Byte;

			switch (ST018.command & 0xFFFFFF)
			{
				case 0x0100: ST018.in_count        = 0;    break;
				case 0xFF00: ST018.in_count        = 0;    break;
				default:     ST018.waiting4command = true; break;
			}
		}
		else
		if (ST018.waiting4command)
		{
			// 3-byte commands
			ST018.part_command++;
			ST018.command <<= 8;
			ST018.command |= Byte;
		}
	}
	// extra parameters
	else
	if (address == 0x3802)
	{
		ST018.parameters[ST018.in_index] = Byte;
		ST018.in_index++;
	}

	if (ST018.in_count == ST018.in_index)
	{
		// qctually execute the command
		ST018.waiting4command = true;
		ST018.in_index        = 0;
		ST018.out_index       = 0;

		switch (ST018.command)
		{
			// hardware check?
			case 0x0100:
				ST018.waiting4command = false;
				ST018.pass++;

				if (ST018.pass == 1)
				{
					ST018.in_count  = 1;
					ST018.out_count = 2;

					// Overload's research
					ST018.output[0x00] = 0x81;
					ST018.output[0x01] = 0x81;
				}
				else
				{
					//ST018.in_count = 1;
					ST018.out_count = 3;

					// no reason to change this
					//ST018.output[0x00] = 0x81;
					//ST018.output[0x01] = 0x81;
					ST018.output[0x02] = 0x81;

					// done processing requests
					if (ST018.pass == 3)
						ST018.waiting4command = true;
				}

				break;

			// unknown: feels like a security detection
			// format identical to 0x0100
			case 0xFF00:
				ST018.waiting4command = false;
				ST018.pass++;

				if (ST018.pass == 1)
				{
					ST018.in_count  = 1;
					ST018.out_count = 2;

					// Overload's research
					ST018.output[0x00] = 0x81;
					ST018.output[0x01] = 0x81;
				}
				else
				{
					//ST018.in_count = 1;
					ST018.out_count = 3;

					// no reason to change this
					//ST018.output[0x00] = 0x81;
					//ST018.output[0x01] = 0x81;
					ST018.output[0x02] = 0x81;

					// done processing requests
					if (ST018.pass == 3)
						ST018.waiting4command = true;
				}

				break;
		}
	}
}
