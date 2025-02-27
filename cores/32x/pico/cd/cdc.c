/***************************************************************************************
 *  Genesis Plus
 *  CD data controller (LC89510 compatible)
 *
 *  Copyright (C) 2012  Eke-Eke (Genesis Plus GX)
 *
 *  Redistribution and use of this code or any derivative works are permitted
 *  provided that the following conditions are met:
 *
 *   - Redistributions may not be sold, nor may they be used in a commercial
 *     product or activity.
 *
 *   - Redistributions that are modified from the original source must include the
 *     complete source code, including the source code for all components used by a
 *     binary built from the modified sources. However, as a special exception, the
 *     source code distributed need not include anything that is normally distributed
 *     (in either source or binary form) with the major components (compiler, kernel,
 *     and so on) of the operating system on which the executable runs, unless that
 *     component itself accompanies the executable.
 *
 *   - Redistributions must reproduce the above copyright notice, this list of
 *     conditions and the following disclaimer in the documentation and/or other
 *     materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************************/

#include "../pico_int.h"
#include "genplus_macros.h"

/* IFSTAT register bitmasks */
#define BIT_DTEI  0x40
#define BIT_DECI  0x20
#define BIT_DTBSY 0x08
#define BIT_DTEN  0x02

/* IFCTRL register bitmasks */
#define BIT_DTEIEN  0x40
#define BIT_DECIEN  0x20
#define BIT_DOUTEN  0x02

/* CTRL0 register bitmasks */
#define BIT_DECEN   0x80
#define BIT_E01RQ   0x20
#define BIT_AUTORQ  0x10
#define BIT_WRRQ    0x04

/* CTRL1 register bitmasks */
#define BIT_MODRQ   0x08
#define BIT_FORMRQ  0x04
#define BIT_SHDREN  0x01

/* CTRL2 register bitmask */
#define BIT_VALST   0x80

/* PicoDrive: doing DMA at once, not using callbacks */
//#define DMA_BYTES_PER_LINE 512

enum dma_type {
  word_ram_0_dma_w = 1,
  word_ram_1_dma_w = 2,
  word_ram_2M_dma_w = 3,
  pcm_ram_dma_w = 4,
  prg_ram_dma_w = 5,
};

/* CDC hardware */
typedef struct
{
  uint8 ifstat;
  uint8 ifctrl;
  uint16 dbc;
  uint16 dac;
  uint16 pt;
  uint16 wa;
  uint8 ctrl[2];
  uint8 head[2][4];
  uint8 stat[4];
  int cycles;
  //void (*dma_w)(unsigned int words);
  int dma_w;
  uint8 ram[0x4000 + 2352]; /* 16K external RAM (with one block overhead to handle buffer overrun) */
} cdc_t; 

static cdc_t cdc;

void cdc_init(void)
{
  memset(&cdc, 0, sizeof(cdc_t));
}

void cdc_reset(void)
{
  /* reset CDC register index */
  Pico_mcd->s68k_regs[0x04+1] = 0x00;

  /* reset CDC registers */
  cdc.ifstat  = 0xff;
  cdc.ifctrl  = 0x00;
  cdc.ctrl[0] = 0x00;
  cdc.ctrl[1] = 0x00;
  cdc.stat[0] = 0x00;
  cdc.stat[1] = 0x00;
  cdc.stat[2] = 0x00;
  cdc.stat[3] = 0x80;
  cdc.head[0][0] = 0x00;
  cdc.head[0][1] = 0x00;
  cdc.head[0][2] = 0x00;
  cdc.head[0][3] = 0x01;
  cdc.head[1][0] = 0x00;
  cdc.head[1][1] = 0x00;
  cdc.head[1][2] = 0x00;
  cdc.head[1][3] = 0x00;

  /* reset CDC cycle counter */
  cdc.cycles = 0;

  /* DMA transfer disabled */
  cdc.dma_w = 0;
}

int cdc_context_save(uint8 *state)
{
  uint8 tmp8;
  int bufferptr = 0;

  if (cdc.dma_w == pcm_ram_dma_w)
  {
    tmp8 = 1;
  }
  else if (cdc.dma_w == prg_ram_dma_w)
  {
    tmp8 = 2;
  }
  else if (cdc.dma_w == word_ram_0_dma_w)
  {
    tmp8 = 3;
  }
  else if (cdc.dma_w == word_ram_1_dma_w)
  {
    tmp8 = 4;
  }
  else if (cdc.dma_w == word_ram_2M_dma_w)
  {
    tmp8 = 5;
  }
  else
  {
    tmp8 = 0;
  }

  save_param(&cdc, sizeof(cdc));
  save_param(&tmp8, 1);

  return bufferptr;
}

int cdc_context_load(uint8 *state)
{
  uint8 tmp8;
  int bufferptr = 0;

  load_param(&cdc, sizeof(cdc));
  load_param(&tmp8, 1);

  switch (tmp8)
  {
    case 1:
      cdc.dma_w = pcm_ram_dma_w;
      break;
    case 2:
      cdc.dma_w = prg_ram_dma_w;
      break;
    case 3:
      cdc.dma_w = word_ram_0_dma_w;
      break;
    case 4:
      cdc.dma_w = word_ram_1_dma_w;
      break;
    case 5:
      cdc.dma_w = word_ram_2M_dma_w;
      break;
    default:
      cdc.dma_w = 0;
      break;
  }

  return bufferptr;
}

int cdc_context_load_old(uint8 *state)
{
#define old_load(v, ofs) \
  memcpy(&cdc.v, state + ofs, sizeof(cdc.v))

  memcpy(cdc.ram, state, 0x4000);
  old_load(ifstat, 67892);
  old_load(ifctrl, 67924);
  old_load(dbc, 67896);
  old_load(dac, 67900);
  old_load(pt, 67908);
  old_load(wa, 67912);
  old_load(ctrl, 67928);
  old_load(head[0], 67904);
  old_load(stat, 67916);

  cdc.dma_w = 0;
  switch (Pico_mcd->s68k_regs[0x04+0] & 0x07)
  {
    case 4: /* PCM RAM DMA */
      cdc.dma_w = pcm_ram_dma_w;
      break;
    case 5: /* PRG-RAM DMA */
      cdc.dma_w = prg_ram_dma_w;
      break;
    case 7: /* WORD-RAM DMA */
      if (Pico_mcd->s68k_regs[0x02+1] & 0x04)
      {
        if (Pico_mcd->s68k_regs[0x02+1] & 0x01)
          cdc.dma_w = word_ram_0_dma_w;
        else
          cdc.dma_w = word_ram_1_dma_w;
      }
      else
      {
        if (Pico_mcd->s68k_regs[0x02+1] & 0x02)
          cdc.dma_w = word_ram_2M_dma_w;
      }
      break;
  }

  return 0x10960; // sizeof(old_cdc)
#undef old_load
}

static void do_dma(enum dma_type type, int bytes_in)
{
  int dma_addr = (Pico_mcd->s68k_regs[0x0a] << 8) | Pico_mcd->s68k_regs[0x0b];
  int src_addr = cdc.dac & 0x3ffe;
  int dst_addr = dma_addr;
  int bytes = bytes_in;
  int words = bytes_in >> 1;
  int dst_limit = 0;
  uint8 *dst;
  int len;

  elprintf(EL_CD, "dma %d %04x->%04x %x",
    type, cdc.dac, dst_addr, bytes_in);

  switch (type)
  {
    case pcm_ram_dma_w:
      dst_addr = (dst_addr << 2) & 0xffc;
      if (dst_addr + bytes > 0x1000) {
        elprintf(EL_ANOMALY, "pcm dma oflow: %x %x", dst_addr, words);
        bytes = 0x1000 - dst_addr;
      }
      dst = Pico_mcd->pcm_ram_b[Pico_mcd->pcm.bank];
      dst = dst + dst_addr;
      while (bytes > 0)
      {
        if (src_addr + bytes > 0x4000) {
          len = 0x4000 - src_addr;
          memcpy(dst, cdc.ram + src_addr, len);
          dst += len;
          src_addr = 0;
          bytes -= len;
          continue;
        }
        memcpy(dst, cdc.ram + src_addr, bytes);
        break;
      }
      goto update_dma;

    case prg_ram_dma_w:
      dst_addr <<= 3;
      dst = Pico_mcd->prg_ram + dst_addr;
      dst_limit = 0x80000;
      break;

    case word_ram_0_dma_w:
      dst_addr = (dst_addr << 3) & 0x1fffe;
      dst = Pico_mcd->word_ram1M[0] + dst_addr;
      dst_limit = 0x20000;
      break;

    case word_ram_1_dma_w:
      dst_addr = (dst_addr << 3) & 0x1fffe;
      dst = Pico_mcd->word_ram1M[1] + dst_addr;
      dst_limit = 0x20000;
      break;

    case word_ram_2M_dma_w:
      dst_addr = (dst_addr << 3) & 0x3fffe;
      dst = Pico_mcd->word_ram2M + dst_addr;
      dst_limit = 0x40000;
      break;

    default:
      elprintf(EL_ANOMALY, "invalid dma: %d", type);
      goto update_dma;
  }

  if (dst_addr + words * 2 > dst_limit) {
    elprintf(EL_ANOMALY, "cd dma %d oflow: %x %x", type, dst_addr, words);
    words = (dst_limit - dst_addr) / 2;
  }
  while (words > 0)
  {
    if (src_addr + words * 2 > 0x4000) {
      len = 0x4000 - src_addr;
      memcpy16bswap((void *)dst, cdc.ram + src_addr, len / 2);
      dst += len;
      src_addr = 0;
      words -= len / 2;
      continue;
    }
    memcpy16bswap((void *)dst, cdc.ram + src_addr, words);
    break;
  }

  bytes_in &= ~1; // Todo leftover byte?

update_dma:
  /* update DMA addresses */
  cdc.dac += bytes_in;
  if (type == pcm_ram_dma_w)
    dma_addr += bytes_in >> 2;
  else
    dma_addr += bytes_in >> 3;

  Pico_mcd->s68k_regs[0x0a] = dma_addr >> 8;
  Pico_mcd->s68k_regs[0x0b] = dma_addr;
}

void cdc_dma_update(void)
{
  /* end of DMA transfer ? */
  //if (cdc.dbc < DMA_BYTES_PER_LINE)
  {
    /* transfer remaining words using 16-bit DMA */
    //cdc.dma_w((cdc.dbc + 1) >> 1);
    do_dma(cdc.dma_w, cdc.dbc + 1);

    /* reset data byte counter (DBCH bits 4-7 should be set to 1) */
    cdc.dbc = 0xf000;

    /* clear !DTEN and !DTBSY */
    cdc.ifstat |= (BIT_DTBSY | BIT_DTEN);

    /* clear DSR bit & set EDT bit (SCD register $04) */
    Pico_mcd->s68k_regs[0x04+0] = (Pico_mcd->s68k_regs[0x04+0] & 0x07) | 0x80;

    if (cdc.ifstat & BIT_DTEI) {
      /* pending Data Transfer End interrupt */
      cdc.ifstat &= ~BIT_DTEI;

      /* Data Transfer End interrupt enabled ? */
      if (cdc.ifctrl & BIT_DTEIEN)
      {
        /* level 5 interrupt enabled ? */
        if (Pico_mcd->s68k_regs[0x32+1] & PCDS_IEN5)
        {
          /* update IRQ level */
          elprintf(EL_INTS, "cdc DTE irq 5");
          pcd_irq_s68k(5, 1);
        }
      }
    }

    /* disable DMA transfer */
    cdc.dma_w = 0;
  }
#if 0
  else
  {
    /* transfer all words using 16-bit DMA */
    cdc.dma_w(DMA_BYTES_PER_LINE >> 1);

    /* decrement data byte counter */
    cdc.dbc -= length;
  }
#endif
}

int cdc_decoder_update(uint8 header[4])
{
  /* data decoding enabled ? */
  if (cdc.ctrl[0] & BIT_DECEN)
  {
    /* update HEAD registers */
    memcpy(cdc.head[0], header, sizeof(cdc.head[0]));

    /* set !VALST */
    cdc.stat[3] = 0x00;

    /* pending decoder interrupt */
    cdc.ifstat &= ~BIT_DECI;

    /* decoder interrupt enabled ? */
    if (cdc.ifctrl & BIT_DECIEN)
    {
      /* level 5 interrupt enabled ? */
      if (Pico_mcd->s68k_regs[0x32+1] & PCDS_IEN5)
      {
        /* update IRQ level */
        elprintf(EL_INTS, "cdc DEC irq 5");
        pcd_irq_s68k(5, 1);
      }
    }

    /* buffer RAM write enabled ? */
    if (cdc.ctrl[0] & BIT_WRRQ)
    {
      uint16 offset;

      /* increment block pointer  */
      cdc.pt += 2352;

      /* increment write address */
      cdc.wa += 2352;

      /* CDC buffer address */
      offset = cdc.pt & 0x3fff;

      /* write CDD block header (4 bytes) */
      memcpy(cdc.ram + offset, header, 4);

      /* write CDD block data (2048 bytes) */
      cdd_read_data(cdc.ram + 4 + offset);

      /* take care of buffer overrun */
      if (offset > (0x4000 - 2048 - 4))
      {
        /* data should be written at the start of buffer */
        memcpy(cdc.ram, cdc.ram + 0x4000, offset + 2048 + 4 - 0x4000);
      }

      /* read next data block */
      return 1;
    }
  }
  
  /* keep decoding same data block if Buffer Write is disabled */
  return 0;
}

void cdc_reg_w(unsigned char data)
{
#ifdef LOG_CDC
  elprintf(EL_STATUS, "CDC register %X write 0x%04x", Pico_mcd->s68k_regs[0x04+1] & 0x0F, data);
#endif
  switch (Pico_mcd->s68k_regs[0x04+1] & 0x1F)
  {
    case 0x00:
      break;

    case 0x01:  /* IFCTRL */
    {
      /* pending interrupts ? */
      if (((data & BIT_DTEIEN) && !(cdc.ifstat & BIT_DTEI)) ||
          ((data & BIT_DECIEN) && !(cdc.ifstat & BIT_DECI)))
      {
        /* level 5 interrupt enabled ? */
        if (Pico_mcd->s68k_regs[0x32+1] & PCDS_IEN5)
        {
          /* update IRQ level */
          elprintf(EL_INTS, "cdc pending irq 5");
          pcd_irq_s68k(5, 1);
        }
      }
      else // if (scd.pending & (1 << 5))
      {
        /* clear pending level 5 interrupts */
        pcd_irq_s68k(5, 0);
      }

      /* abort any data transfer if data output is disabled */
      if (!(data & BIT_DOUTEN))
      {
        /* clear !DTBSY and !DTEN */
        cdc.ifstat |= (BIT_DTBSY | BIT_DTEN);
      }

      cdc.ifctrl = data;
      Pico_mcd->s68k_regs[0x04+1] = 0x02;
      break;
    }

    case 0x02:  /* DBCL */
      cdc.dbc &= 0xff00;
      cdc.dbc |= data;
      Pico_mcd->s68k_regs[0x04+1] = 0x03;
      break;

    case 0x03:  /* DBCH */
      cdc.dbc &= 0x00ff;
      cdc.dbc |= (data & 0x0f) << 8;
      Pico_mcd->s68k_regs[0x04+1] = 0x04;
      break;

    case 0x04:  /* DACL */
      cdc.dac &= 0xff00;
      cdc.dac |= data;
      Pico_mcd->s68k_regs[0x04+1] = 0x05;
      break;

    case 0x05:  /* DACH */
      cdc.dac &= 0x00ff;
      cdc.dac |= data << 8;
      Pico_mcd->s68k_regs[0x04+1] = 0x06;
      break;

    case 0x06:  /* DTRG */
    {
      /* start data transfer if data output is enabled */
      if (cdc.ifctrl & BIT_DOUTEN)
      {
        /* set !DTBSY */
        cdc.ifstat &= ~BIT_DTBSY;

        /* clear DBCH bits 4-7 */
        cdc.dbc &= 0x0fff;

        /* clear EDT & DSR bits (SCD register $04) */
        Pico_mcd->s68k_regs[0x04+0] &= 0x07;

        cdc.dma_w = 0;

        /* setup data transfer destination */
        switch (Pico_mcd->s68k_regs[0x04+0] & 0x07)
        {
          case 2: /* MAIN-CPU host read */
          case 3: /* SUB-CPU host read */
          {
            /* set !DTEN */
            cdc.ifstat &= ~BIT_DTEN;

            /* set DSR bit (register $04) */
            Pico_mcd->s68k_regs[0x04+0] |= 0x40;
            break;
          }

          case 4: /* PCM RAM DMA */
          {
            cdc.dma_w = pcm_ram_dma_w;
            break;
          }

          case 5: /* PRG-RAM DMA */
          {
            cdc.dma_w = prg_ram_dma_w;
            break;
          }

          case 7: /* WORD-RAM DMA */
          {
            /* check memory mode */
            if (Pico_mcd->s68k_regs[0x02+1] & 0x04)
            {
              /* 1M mode */
              if (Pico_mcd->s68k_regs[0x02+1] & 0x01)
              {
                /* Word-RAM bank 0 is assigned to SUB-CPU */
                cdc.dma_w = word_ram_0_dma_w;
              }
              else
              {
                /* Word-RAM bank 1 is assigned to SUB-CPU */
                cdc.dma_w = word_ram_1_dma_w;
              }
            }
            else
            {
              /* 2M mode */
              if (Pico_mcd->s68k_regs[0x02+1] & 0x02)
              {
                /* only process DMA if Word-RAM is assigned to SUB-CPU */
                cdc.dma_w = word_ram_2M_dma_w;
              }
            }
            break;
          }

          default: /* invalid */
          {
            elprintf(EL_ANOMALY, "invalid CDC tranfer destination (%d)",
              Pico_mcd->s68k_regs[0x04+0] & 0x07);
            break;
          }
        }

        if (cdc.dma_w)
          pcd_event_schedule_s68k(PCD_EVENT_DMA, cdc.dbc / 2);
      }

      Pico_mcd->s68k_regs[0x04+1] = 0x07;
      break;
    }

    case 0x07:  /* DTACK */
    {
      /* clear pending data transfer end interrupt */
      cdc.ifstat |= BIT_DTEI;

      /* clear DBCH bits 4-7 */
      cdc.dbc &= 0x0fff;

#if 0
      /* no pending decoder interrupt ? */
      if ((cdc.ifstat | BIT_DECI) || !(cdc.ifctrl & BIT_DECIEN))
      {
        /* clear pending level 5 interrupt */
        pcd_irq_s68k(5, 0);
      }
#endif
      Pico_mcd->s68k_regs[0x04+1] = 0x08;
      break;
    }

    case 0x08:  /* WAL */
      cdc.wa &= 0xff00;
      cdc.wa |= data;
      Pico_mcd->s68k_regs[0x04+1] = 0x09;
      break;

    case 0x09:  /* WAH */
      cdc.wa &= 0x00ff;
      cdc.wa |= data << 8;
      Pico_mcd->s68k_regs[0x04+1] = 0x0a;
      break;

    case 0x0a:  /* CTRL0 */
    {
      /* set CRCOK bit only if decoding is enabled */
      cdc.stat[0] = data & BIT_DECEN;

      /* reset DECI if decoder turned off */
      if (!cdc.stat[0])
        cdc.ifstat |= BIT_DECI;

      /* update decoding mode */
      if (data & BIT_AUTORQ)
      {
        /* set MODE bit according to CTRL1 register & clear FORM bit */
        cdc.stat[2] = cdc.ctrl[1] & BIT_MODRQ;
      }
      else 
      {
        /* set MODE & FORM bits according to CTRL1 register */
        cdc.stat[2] = cdc.ctrl[1] & (BIT_MODRQ | BIT_FORMRQ);
      }

      cdc.ctrl[0] = data;
      Pico_mcd->s68k_regs[0x04+1] = 0x0b;
      break;
    }

    case 0x0b:  /* CTRL1 */
    {
      /* update decoding mode */
      if (cdc.ctrl[0] & BIT_AUTORQ)
      {
        /* set MODE bit according to CTRL1 register & clear FORM bit */
        cdc.stat[2] = data & BIT_MODRQ;
      }
      else 
      {
        /* set MODE & FORM bits according to CTRL1 register */
        cdc.stat[2] = data & (BIT_MODRQ | BIT_FORMRQ);
      }

      cdc.ctrl[1] = data;
      Pico_mcd->s68k_regs[0x04+1] = 0x0c;
      break;
    }

    case 0x0c:  /* PTL */
      cdc.pt &= 0xff00;
      cdc.pt |= data;
      Pico_mcd->s68k_regs[0x04+1] = 0x0d;
      break;
  
    case 0x0d:  /* PTH */
      cdc.pt &= 0x00ff;
      cdc.pt |= data << 8;
      Pico_mcd->s68k_regs[0x04+1] = 0x0e;
      break;

    case 0x0e:  /* CTRL2 (unused) */
      Pico_mcd->s68k_regs[0x04+1] = 0x0f;
      break;

    case 0x0f:  /* RESET */
      cdc_reset();
      break;

    default:  /* by default, SBOUT is not used */
      Pico_mcd->s68k_regs[0x04+1] = (Pico_mcd->s68k_regs[0x04+1] + 1) & 0x1f;
      break;
  }
}

unsigned char cdc_reg_r(void)
{
  switch (Pico_mcd->s68k_regs[0x04+1] & 0x1F)
  {
    case 0x00:
      return 0xff;

    case 0x01:  /* IFSTAT */
      Pico_mcd->s68k_regs[0x04+1] = 0x02;
      return cdc.ifstat;

    case 0x02:  /* DBCL */
      Pico_mcd->s68k_regs[0x04+1] = 0x03;
      return cdc.dbc & 0xff;

    case 0x03:  /* DBCH */
      Pico_mcd->s68k_regs[0x04+1] = 0x04;
      return (cdc.dbc >> 8) & 0xff;

    case 0x04:  /* HEAD0 */
      Pico_mcd->s68k_regs[0x04+1] = 0x05;
      return cdc.head[cdc.ctrl[1] & BIT_SHDREN][0];

    case 0x05:  /* HEAD1 */
      Pico_mcd->s68k_regs[0x04+1] = 0x06;
      return cdc.head[cdc.ctrl[1] & BIT_SHDREN][1];

    case 0x06:  /* HEAD2 */
      Pico_mcd->s68k_regs[0x04+1] = 0x07;
      return cdc.head[cdc.ctrl[1] & BIT_SHDREN][2];

    case 0x07:  /* HEAD3 */
      Pico_mcd->s68k_regs[0x04+1] = 0x08;
      return cdc.head[cdc.ctrl[1] & BIT_SHDREN][3];

    case 0x08:  /* PTL */
      Pico_mcd->s68k_regs[0x04+1] = 0x09;
      return cdc.pt & 0xff;

    case 0x09:  /* PTH */
      Pico_mcd->s68k_regs[0x04+1] = 0x0a;
      return (cdc.pt >> 8) & 0xff;

    case 0x0a:  /* WAL */
      Pico_mcd->s68k_regs[0x04+1] = 0x0b;
      return cdc.wa & 0xff;

    case 0x0b:  /* WAH */
      Pico_mcd->s68k_regs[0x04+1] = 0x0c;
      return (cdc.wa >> 8) & 0xff;

    case 0x0c: /* STAT0 */
      Pico_mcd->s68k_regs[0x04+1] = 0x0d;
      return cdc.stat[0];

    case 0x0d: /* STAT1 (always return 0) */
      Pico_mcd->s68k_regs[0x04+1] = 0x0e;
      return 0x00;

    case 0x0e:  /* STAT2 */
      Pico_mcd->s68k_regs[0x04+1] = 0x0f;
      return cdc.stat[2];

    case 0x0f:  /* STAT3 */
    {
      uint8 data = cdc.stat[3];

      /* clear !VALST (note: this is not 100% correct but BIOS do not seem to care) */
      cdc.stat[3] = BIT_VALST;

      /* clear pending decoder interrupt */
      cdc.ifstat |= BIT_DECI;
      
#if 0
      /* no pending data transfer end interrupt */
      if ((cdc.ifstat | BIT_DTEI) || !(cdc.ifctrl & BIT_DTEIEN))
      {
        /* clear pending level 5 interrupt */
        pcd_irq_s68k(5, 0);
      }
#endif

      Pico_mcd->s68k_regs[0x04+1] = 0x10;
      return data;
    }

    default:  /* by default, COMIN is always empty */
      Pico_mcd->s68k_regs[0x04+1] = (Pico_mcd->s68k_regs[0x04+1] + 1) & 0x1f;
      return 0xff;
  }
}

unsigned short cdc_host_r(void)
{
  /* check if data is available */
  if (!(cdc.ifstat & BIT_DTEN))
  {
    /* read data word from CDC RAM buffer */
    uint8 *datap = cdc.ram + (cdc.dac & 0x3ffe);
    uint16 data = (datap[0] << 8) | datap[1];

#ifdef LOG_CDC
    error("CDC host read 0x%04x -> 0x%04x (dbc=0x%x) (%X)\n", cdc.dac, data, cdc.dbc, s68k.pc);
#endif
 
    /* increment data address counter */
    cdc.dac += 2;

    /* decrement data byte counter */
    cdc.dbc -= 2;

    /* end of transfer ? */
    if ((int16)cdc.dbc <= 0)
    {
      /* reset data byte counter (DBCH bits 4-7 should be set to 1) */
      cdc.dbc = 0xf000;

      /* clear !DTEN and !DTBSY */
      cdc.ifstat |= (BIT_DTBSY | BIT_DTEN);

      /* clear DSR bit & set EDT bit (SCD register $04) */
      Pico_mcd->s68k_regs[0x04+0] = (Pico_mcd->s68k_regs[0x04+0] & 0x07) | 0x80;

    } else if ((int16)cdc.dbc <= 2)
    {
      if (cdc.ifstat & BIT_DTEI) {
        /* pending Data Transfer End interrupt */
        cdc.ifstat &= ~BIT_DTEI;

        /* Data Transfer End interrupt enabled ? */
        if (cdc.ifctrl & BIT_DTEIEN)
        {
          /* level 5 interrupt enabled ? */
          if (Pico_mcd->s68k_regs[0x32+1] & PCDS_IEN5)
          {
            /* update IRQ level */
            elprintf(EL_INTS, "cdc DTE irq 5");
            pcd_irq_s68k(5, 1);
          }
        }
      }
      /* set DSR and EDT bit (SCD register $04) */
      Pico_mcd->s68k_regs[0x04+0] = (Pico_mcd->s68k_regs[0x04+0] & 0x07) | 0xc0;
    }

    return data;
  }

#ifdef LOG_CDC
  error("error reading CDC host (data transfer disabled)\n");
#endif
  return 0xffff;
}

// vim:shiftwidth=2:ts=2:expandtab
