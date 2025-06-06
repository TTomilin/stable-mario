/******************************************************************************/
/* Mednafen Sega Saturn Emulation Module                                      */
/******************************************************************************/
/* sound.cpp - Sound Emulation
**  Copyright (C) 2015-2021 Mednafen Team
**
** This program is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License
** as published by the Free Software Foundation; either version 2
** of the License, or (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software Foundation, Inc.,
** 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

// TODO: Bus between SCU and SCSP looks to be 8-bit, maybe implement that, but
// first test to see how the bus access cycle(s) work with respect to reading from
// registers whose values may change between the individual byte reads.
// (May not be worth emulating if it could possibly trigger problems in games)

#include "../mednafen.h"
#include "../hw_cpu/m68k/m68k.h"
#include "../jump.h"

#include "ss.h"
#include "sound.h"
#include "scu.h"
#include "cdb.h"

#include "scsp.h"

static SS_SCSP SCSP;

static M68K SoundCPU(true);
static int64 run_until_time;	// 32.32
static int32 next_scsp_time;

static uint32 clock_ratio;
static sscpu_timestamp_t lastts;

static MDFN_jmp_buf jbuf;

int16 IBuffer[1024][2];
uint32 IBufferCount;

static INLINE void SCSP_SoundIntChanged(SS_SCSP* s, unsigned level)
{
 SoundCPU.SetIPL(level);
}

static INLINE void SCSP_MainIntChanged(SS_SCSP* s, bool state)
{
 SCU_SetInt(SCU_INT_SCSP, state);
}

#include "scsp.inc"

//
//
template<typename T>
static MDFN_FASTCALL T SoundCPU_BusRead(uint32 A);

static MDFN_FASTCALL uint16 SoundCPU_BusReadInstr(uint32 A);

template<typename T>
static MDFN_FASTCALL void SoundCPU_BusWrite(uint32 A, T V);

static MDFN_FASTCALL void SoundCPU_BusRMW(uint32 A, uint8 (MDFN_FASTCALL *cb)(M68K*, uint8));
static MDFN_FASTCALL unsigned SoundCPU_BusIntAck(uint8 level);
static MDFN_FASTCALL void SoundCPU_BusRESET(bool state);
//
//

void SOUND_Init(void)
{
 memset(IBuffer, 0, sizeof(IBuffer));
 IBufferCount = 0;

 run_until_time = 0;
 next_scsp_time = 0;
 lastts = 0;

 SoundCPU.BusRead8 = SoundCPU_BusRead<uint8>;
 SoundCPU.BusRead16 = SoundCPU_BusRead<uint16>;

 SoundCPU.BusWrite8 = SoundCPU_BusWrite<uint8>;
 SoundCPU.BusWrite16 = SoundCPU_BusWrite<uint16>;

 SoundCPU.BusReadInstr = SoundCPU_BusReadInstr;

 SoundCPU.BusRMW = SoundCPU_BusRMW;

 SoundCPU.BusIntAck = SoundCPU_BusIntAck;
 SoundCPU.BusRESET = SoundCPU_BusRESET;

 SS_SetPhysMemMap(0x05A00000, 0x05A7FFFF, SCSP.GetRAMPtr(), 0x80000, true);
 // TODO: MEM4B: SS_SetPhysMemMap(0x05A00000, 0x05AFFFFF, SCSP.GetRAMPtr(), 0x40000, true);
}

uint8 SOUND_PeekRAM(uint32 A)
{
 return ne16_rbo_be<uint8>(SCSP.GetRAMPtr(), A & 0x7FFFF);
}

void SOUND_PokeRAM(uint32 A, uint8 V)
{
 ne16_wbo_be<uint8>(SCSP.GetRAMPtr(), A & 0x7FFFF, V);
}

static INLINE void ResetTS_68K(void)
{
 next_scsp_time -= SoundCPU.timestamp;
 run_until_time -= (int64)SoundCPU.timestamp << 32;
 SoundCPU.timestamp = 0;
}

void SOUND_AdjustTS(const int32 delta)
{
 ResetTS_68K();
 //
 //
 lastts += delta;
}

void SOUND_Reset(bool powering_up)
{
 SCSP.Reset(powering_up);
 SoundCPU.Reset(powering_up);
}

void SOUND_Reset68K(void)
{
 SoundCPU.Reset(false);
}

void SOUND_Kill(void)
{
}

void SOUND_Set68KActive(bool active)
{
 SoundCPU.SetExtHalted(!active);
}

uint16 SOUND_Read16(uint32 A)
{
 uint16 ret;

 SCSP.RW<uint16, false>(A, ret);

 return ret;
}

void SOUND_Write8(uint32 A, uint8 V)
{
 SCSP.RW<uint8, true>(A, V);
}

void SOUND_Write16(uint32 A, uint16 V)
{
 SCSP.RW<uint16, true>(A, V);
}

static NO_INLINE void RunSCSP(void)
{
 CDB_GetCDDA(SCSP.GetEXTSPtr());
 //
 //
 int16* const bp = IBuffer[IBufferCount];
 SCSP.RunSample(bp);
 //bp[0] = rand();
 //bp[1] = rand();
 bp[0] = (bp[0] * 27 + 16) >> 5;
 bp[1] = (bp[1] * 27 + 16) >> 5;

/*
 // TODO?  Need to measure frequency response more reliably first, ideally after capacitor
 // replacement.  Should probably be controlled by a boolean setting, too.
 for(unsigned lr = 0; lr < 2; lr++)
 {
  static int32 filt[2];
  filt[lr] += (((int64)(int32)((uint32)bp[lr] << 16) - filt[lr]) * 60500) >> 16;
  bp[lr] = filt[lr] >> 16;
 }
*/

 IBufferCount = (IBufferCount + 1) & 1023;
 next_scsp_time += 256;
}

// Ratio between SH-2 clock and 68K clock (sound clock / 2)
void SOUND_SetClockRatio(uint32 ratio)
{
 clock_ratio = ratio;
}

sscpu_timestamp_t SOUND_Update(sscpu_timestamp_t timestamp)
{
 run_until_time += ((uint64)(timestamp - lastts) * clock_ratio);
 lastts = timestamp;
 //
 //
 MDFN_setjmp(jbuf);

 if(MDFN_LIKELY(SoundCPU.timestamp < (run_until_time >> 32)))
 {
  do
  {
   int32 next_time = std::min<int32>(next_scsp_time, run_until_time >> 32);

   SoundCPU.Run(next_time);

   if(SoundCPU.timestamp >= next_scsp_time)
    RunSCSP();
  } while(MDFN_LIKELY(SoundCPU.timestamp < (run_until_time >> 32)));
 }
 else
 {
  while(next_scsp_time < (run_until_time >> 32))
   RunSCSP();
 }

 return timestamp + 128;	// FIXME
}

void SOUND_StartFrame(double rate, uint32 quality)
{
}

void SOUND_StateAction(StateMem* sm, const unsigned load, const bool data_only)
{
 SFORMAT StateRegs[] =
 {
  SFVAR(next_scsp_time),
  SFVAR(run_until_time),

  SFEND
 };

 //
 next_scsp_time -= SoundCPU.timestamp;
 run_until_time -= (int64)SoundCPU.timestamp << 32;

 MDFNSS_StateAction(sm, load, data_only, StateRegs, "SOUND", false);

 next_scsp_time += SoundCPU.timestamp;
 run_until_time += (int64)SoundCPU.timestamp << 32;
 //

 SoundCPU.StateAction(sm, load, data_only, "M68K");
 SCSP.StateAction(sm, load, data_only, "SCSP");
}

//
//
//
template<typename T>
static MDFN_FASTCALL T SoundCPU_BusRead(uint32 A)
{
 if(MDFN_UNLIKELY(A & (0xE00000 | (sizeof(T) - 1))))
 {
  SoundCPU.timestamp += 4;

  if(A & (sizeof(T) - 1))
   SoundCPU.SignalAddressError(A, 0x3);
  else
   SoundCPU.SignalDTACKHalted(A);

  MDFN_longjmp(jbuf);
 }
 //
 T ret;

 SoundCPU.timestamp += 4;

 if(MDFN_UNLIKELY(SoundCPU.timestamp >= next_scsp_time))
  RunSCSP();

 SCSP.RW<T, false>(A & 0x1FFFFF, ret);

 SoundCPU.timestamp += 2;

 return ret;
}

static MDFN_FASTCALL uint16 SoundCPU_BusReadInstr(uint32 A)
{
 if(MDFN_UNLIKELY(A & 0xE00001))
 {
  SoundCPU.timestamp += 4;

  if(A & 1)
   SoundCPU.SignalAddressError(A, 0x2);
  else
   SoundCPU.SignalDTACKHalted(A);

  MDFN_longjmp(jbuf);
 }
 //
 uint16 ret;

 SoundCPU.timestamp += 4;

 //if(MDFN_UNLIKELY(SoundCPU.timestamp >= next_scsp_time))
 // RunSCSP();

 SCSP.RW<uint16, false>(A & 0x1FFFFF, ret);

 SoundCPU.timestamp += 2;

 return ret;
}

template<typename T>
static MDFN_FASTCALL void SoundCPU_BusWrite(uint32 A, T V)
{
 if(MDFN_UNLIKELY(A & (0xE00000 | (sizeof(T) - 1))))
 {
  SoundCPU.timestamp += 4;

  if(A & (sizeof(T) - 1))
   SoundCPU.SignalAddressError(A, 0x1);
  else
   SoundCPU.SignalDTACKHalted(A);

  MDFN_longjmp(jbuf);
 }
 //
 SoundCPU.timestamp += 2;

 if(MDFN_UNLIKELY(SoundCPU.timestamp >= next_scsp_time))
  RunSCSP();

 SoundCPU.timestamp += 2;

 SCSP.RW<T, true>(A & 0x1FFFFF, V);
 SoundCPU.timestamp += 2;
}


static MDFN_FASTCALL void SoundCPU_BusRMW(uint32 A, uint8 (MDFN_FASTCALL *cb)(M68K*, uint8))
{
 if(MDFN_UNLIKELY(A & 0xE00000))
 {
  SoundCPU.timestamp += 4;
  SoundCPU.SignalDTACKHalted(A);
  MDFN_longjmp(jbuf);
 }
 //
 uint8 tmp;

 SoundCPU.timestamp += 4;

 if(MDFN_UNLIKELY(SoundCPU.timestamp >= next_scsp_time))
  RunSCSP();

 SCSP.RW<uint8, false>(A & 0x1FFFFF, tmp);

 tmp = cb(&SoundCPU, tmp);

 SoundCPU.timestamp += 6;

 SCSP.RW<uint8, true>(A & 0x1FFFFF, tmp);

 SoundCPU.timestamp += 2;
}

static MDFN_FASTCALL unsigned SoundCPU_BusIntAck(uint8 level)
{
 return M68K::BUS_INT_ACK_AUTO;
}

static MDFN_FASTCALL void SoundCPU_BusRESET(bool state)
{
 if(state)
  SoundCPU.Reset(false);
}

uint32 SOUND_GetSCSPRegister(const unsigned id, char* const special, const uint32 special_len)
{
 return SCSP.GetRegister(id, special, special_len);
}

void SOUND_SetSCSPRegister(const unsigned id, const uint32 value)
{
 SCSP.SetRegister(id, value);
}

uint32 SOUND_GetM68KRegister(const unsigned id, char* const special, const uint32 special_len)
{
 return SoundCPU.GetRegister(id, special, special_len);
}

void SOUND_SetM68KRegister(const unsigned id, const uint32 value)
{
 SoundCPU.SetRegister(id, value);
}
