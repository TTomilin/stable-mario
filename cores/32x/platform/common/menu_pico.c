/*
 * PicoDrive
 * (C) notaz, 2010,2011
 *
 * This work is licensed under the terms of MAME license.
 * See COPYING file in the top-level directory.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "emu.h"
#include "menu_pico.h"
#include "input_pico.h"
#include "version.h"

#include <pico/pico_int.h>
#include <pico/patch.h>

#ifdef PANDORA
#define MENU_X2 1
#else
#define MENU_X2 0
#endif

#if defined USE_BGR555
#define COL_ROM	0x5eff
#define COL_OTH	0x5ff5
#elif defined USE_BGR565
#define COL_ROM	0xfdf7
#define COL_OTH	0xaff5
#else
#define COL_ROM	0xbdff
#define	COL_OTH	0xaff5
#endif

// FIXME
#ifndef REVISION
#define REVISION "0"
#endif

static const char *rom_exts[] = {
	"zip", "bin",
	"pco", "smd", "gen", "md",
	"iso", "cso", "cue", "chd",
	"32x",
	"sms", "gg",  "sg", "sc",
	NULL
};

// rrrr rggg gggb bbbb
static unsigned short fname2color(const char *fname)
{
	static const char *other_exts[] = { "gmv", "pat" };
	const char *ext;
	int i;

	ext = strrchr(fname, '.');
	if (ext++ == NULL) {
		ext = fname + strlen(fname) - 3;
		if (ext < fname) ext = fname;
	}

	for (i = 0; rom_exts[i] != NULL; i++)
		if (strcasecmp(ext, rom_exts[i]) == 0) return COL_ROM;
	for (i = 0; i < array_size(other_exts); i++)
		if (strcasecmp(ext, other_exts[i]) == 0) return COL_OTH;
	return 0xffff;
}

#include <platform/libpicofe/menu.c>

static const char *men_dummy[] = { NULL };
static int menu_w, menu_h;

/* platform specific options and handlers */
#if   defined(__GP2X__)
#include <platform/gp2x/menu.c>
#elif defined(__PSP__)
#include <platform/psp/menu.c>
#elif defined(PANDORA)
#include <platform/pandora/menu.c>
#else
#include <platform/linux/menu.c>
#endif

static void make_bg(int no_scale, int from_screen)
{
	unsigned short *src = (void *)g_menubg_src_ptr;
	int w = g_menubg_src_w ? g_menubg_src_w : g_screen_width;
	int h = g_menubg_src_h ? g_menubg_src_h : g_screen_height;
	int pp = g_menubg_src_pp ? g_menubg_src_pp : g_screen_ppitch;
	short *dst;
	int x, y;

	if (from_screen) {
		src = g_screen_ptr;
		w = g_screen_width;
		h = g_screen_height;
		pp = g_screen_ppitch;
	}

	memset(g_menubg_ptr, 0, g_menuscreen_w * g_menuscreen_h * 2);
	if (src == NULL)
		return;

	if (!no_scale && g_menuscreen_w / w >= 2 && g_menuscreen_h / h >= 2)
	{
		u32 t, *d = g_menubg_ptr;
		d += (g_menuscreen_h / 2 - h * 2 / 2)
			* g_menuscreen_w / 2;
		d += (g_menuscreen_w / 2 - w * 2 / 2) / 2;
		for (y = 0; y < h; y++, src += pp, d += g_menuscreen_w*2/2) {
			for (x = 0; x < w; x++) {
				t = src[x];
				t = ((t & 0xf79e)>>1) - ((t & 0xc618)>>3);
				t |= t << 16;
				d[x] = d[x + g_menuscreen_w / 2] = t;
			}
		}
		return;
	}

	if (w > g_menuscreen_w)
		w = g_menuscreen_w;
	if (h > g_menuscreen_h)
		h = g_menuscreen_h;
	dst = (short *)g_menubg_ptr +
		(g_menuscreen_h / 2 - h / 2) * g_menuscreen_w +
		(g_menuscreen_w / 2 - w / 2);

	// darken the active framebuffer
	for (; h > 0; dst += g_menuscreen_w, src += pp, h--)
		menu_darken_bg(dst, src, w, 1);
}

static void copy_bg(int dir)
{
	unsigned short *bg = (void *)g_menubg_ptr;
	unsigned short *sc = (void *)g_menuscreen_ptr;
	int h = g_menuscreen_h;

	for (; h > 0; sc += g_menuscreen_pp, bg += g_menuscreen_w, h--) {
		if (dir)
			memcpy(bg, sc, g_menuscreen_w * 2);
		else
			memcpy(sc, bg, g_menuscreen_w * 2);
	}
}

static void menu_draw_prep(void)
{
	if (menu_w == g_menuscreen_w && menu_h == g_menuscreen_h)
		return;
	menu_w = g_menuscreen_w, menu_h = g_menuscreen_h;

	if (PicoGameLoaded)
	{
		make_bg(0, 0);
	}
	else
	{
		int pos;
		char buff[256];
		pos = plat_get_skin_dir(buff, 256);
		strcpy(buff + pos, "background.png");

		// should really only happen once, on startup..
		memset(g_menubg_ptr, 0, g_menuscreen_w * g_menuscreen_h * 2);
		if (readpng(g_menubg_ptr, buff, READPNG_BG,
						g_menuscreen_w, g_menuscreen_h) < 0)
			memset(g_menubg_ptr, 0, g_menuscreen_w * g_menuscreen_h * 2);
	}
}

static void draw_savestate_bg(int slot)
{
	const char *fname;
	void *tmp_state;

	fname = emu_get_save_fname(1, 0, slot, NULL);
	if (!fname)
		return;

	tmp_state = PicoTmpStateSave();

	PicoStateLoadGfx(fname);

	/* do a frame and fetch menu bg */
	pemu_forced_frame(0, 0);

	make_bg(0, 1);

	PicoTmpStateRestore(tmp_state);
}

static void menu_enter(int is_rom_loaded)
{
	plat_video_menu_enter(is_rom_loaded);
	menu_w = menu_h = 0;
	menu_draw_prep();
}

// --------- loading ROM screen ----------

static int cdload_called = 0;

static void load_progress_cb(int percent)
{
	int ln, len = percent * g_menuscreen_w / 100;
	unsigned short *dst;

	if (len > g_menuscreen_w)
		len = g_menuscreen_w;

	menu_draw_begin(0, 1);
	copy_bg(0);
	dst = (unsigned short *)g_menuscreen_ptr + g_menuscreen_pp * me_sfont_h * 2;
	for (ln = me_sfont_h - 2; ln > 0; ln--, dst += g_menuscreen_pp)
		memset(dst, 0xff, len * 2);
	menu_draw_end();
}

static void cdload_progress_cb(const char *fname, int percent)
{
	int ln, len = percent * g_menuscreen_w / 100;
	unsigned short *dst;

	menu_draw_begin(0, 1);
	dst = (unsigned short *)g_menuscreen_ptr + g_menuscreen_pp * me_sfont_h * 2;

	copy_bg(0);
	menuscreen_memset_lines(dst, 0xff, me_sfont_h - 2);

	smalltext_out16(1, 3 * me_sfont_h, "Processing CD image / MP3s", 0xffff);
	smalltext_out16(1, 4 * me_sfont_h, fname, 0xffff);
	dst += g_menuscreen_pp * me_sfont_h * 3;

	if (len > g_menuscreen_w)
		len = g_menuscreen_w;

	for (ln = (me_sfont_h - 2); ln > 0; ln--, dst += g_menuscreen_pp)
		memset(dst, 0xff, len * 2);
	menu_draw_end();

	cdload_called = 1;
}

void menu_romload_prepare(const char *rom_name)
{
	const char *p = rom_name + strlen(rom_name);

	while (p > rom_name && *p != '/')
		p--;

	menu_draw_begin(1, 1);
	smalltext_out16(1, 1, "Loading", 0xffff);
	smalltext_out16(1, me_sfont_h, p, 0xffff);
	/* copy menu to bg for callbacks. OK since we are not in menu_loop here */
	copy_bg(1);
	menu_draw_end();

	PicoCartLoadProgressCB = load_progress_cb;
	PicoCDLoadProgressCB = cdload_progress_cb;
	cdload_called = 0;
}

void menu_romload_end(void)
{
	PicoCartLoadProgressCB = NULL;
	PicoCDLoadProgressCB = NULL;

	menu_draw_begin(0, 1);
	copy_bg(0);
	smalltext_out16(1, (cdload_called ? 6 : 3) * me_sfont_h,
		"Starting emulation...", 0xffff);
	menu_draw_end();
}

// ------------ patch/gg menu ------------

static void draw_patchlist(int sel)
{
	int max_cnt, start, i, pos, active;

	max_cnt = g_menuscreen_h / me_sfont_h;
	start = max_cnt / 2 - sel;

	menu_draw_begin(1, 0);

	for (i = 0; i < PicoPatchCount; i++) {
		pos = start + i;
		if (pos < 0) continue;
		if (pos >= max_cnt) break;
		active = PicoPatches[i].active;
		smalltext_out16(14,                pos * me_sfont_h, active ? "ON " : "OFF", active ? 0xfff6 : 0xffff);
		smalltext_out16(14 + me_sfont_w*4, pos * me_sfont_h, PicoPatches[i].name,    active ? 0xfff6 : 0xffff);
	}
	pos = start + i;
	if (pos < max_cnt)
		smalltext_out16(14, pos * me_sfont_h, "done", 0xffff);

	text_out16(5, max_cnt / 2 * me_sfont_h, ">");
	menu_draw_end();
}

static void menu_loop_patches(void)
{
	static int menu_sel = 0;
	int inp;

	for (;;)
	{
		draw_patchlist(menu_sel);
		inp = in_menu_wait(PBTN_UP|PBTN_DOWN|PBTN_LEFT|PBTN_RIGHT|PBTN_L|PBTN_R
				|PBTN_MOK|PBTN_MBACK, NULL, 33);
		if (inp & PBTN_UP  ) { menu_sel--; if (menu_sel < 0) menu_sel = PicoPatchCount; }
		if (inp & PBTN_DOWN) { menu_sel++; if (menu_sel > PicoPatchCount) menu_sel = 0; }
		if (inp &(PBTN_LEFT|PBTN_L))  { menu_sel-=10; if (menu_sel < 0) menu_sel = 0; }
		if (inp &(PBTN_RIGHT|PBTN_R)) { menu_sel+=10; if (menu_sel > PicoPatchCount) menu_sel = PicoPatchCount; }
		if (inp & PBTN_MOK) { // action
			if (menu_sel < PicoPatchCount)
				PicoPatches[menu_sel].active = !PicoPatches[menu_sel].active;
			else 	break;
		}
		if (inp & PBTN_MBACK)
			break;
	}
}

// -------------- key config --------------

// PicoIn.pad[] format: MXYZ SACB RLDU
me_bind_action me_ctrl_actions[] =
{
	{ "UP     ", 0x0001 },
	{ "DOWN   ", 0x0002 },
	{ "LEFT   ", 0x0004 },
	{ "RIGHT  ", 0x0008 },
	{ "A      ", 0x0040 },
	{ "B      ", 0x0010 },
	{ "C      ", 0x0020 },
	{ "A turbo", 0x4000 },
	{ "B turbo", 0x1000 },
	{ "C turbo", 0x2000 },
	{ "START  ", 0x0080 },
	{ "MODE   ", 0x0800 },
	{ "X      ", 0x0400 },
	{ "Y      ", 0x0200 },
	{ "Z      ", 0x0100 },
	{ NULL,      0 },
};

me_bind_action emuctrl_actions[] =
{
	{ "Load State       ", PEV_STATE_LOAD },
	{ "Save State       ", PEV_STATE_SAVE },
	{ "Prev Save Slot   ", PEV_SSLOT_PREV },
	{ "Next Save Slot   ", PEV_SSLOT_NEXT },
	{ "Switch Renderer  ", PEV_SWITCH_RND },
	{ "Volume Down      ", PEV_VOL_DOWN },
	{ "Volume Up        ", PEV_VOL_UP },
	{ "Fast forward     ", PEV_FF },
	{ "Reset Game       ", PEV_RESET },
	{ "Enter Menu       ", PEV_MENU },
	{ "Pico Next page   ", PEV_PICO_PNEXT },
	{ "Pico Prev page   ", PEV_PICO_PPREV },
	{ "Pico Switch input", PEV_PICO_SWINP },
	{ NULL,                0 }
};

static int key_config_loop_wrap(int id, int keys)
{
	switch (id) {
		case MA_CTRL_PLAYER1:
			key_config_loop(me_ctrl_actions, array_size(me_ctrl_actions) - 1, 0);
			break;
		case MA_CTRL_PLAYER2:
			key_config_loop(me_ctrl_actions, array_size(me_ctrl_actions) - 1, 1);
			break;
		case MA_CTRL_PLAYER3:
			key_config_loop(me_ctrl_actions, array_size(me_ctrl_actions) - 1, 2);
			break;
		case MA_CTRL_PLAYER4:
			key_config_loop(me_ctrl_actions, array_size(me_ctrl_actions) - 1, 3);
			break;
		case MA_CTRL_EMU:
			key_config_loop(emuctrl_actions, array_size(emuctrl_actions) - 1, -1);
			break;
		default:
			break;
	}
	return 0;
}

static const char *mgn_dev_name(int id, int *offs)
{
	const char *name = NULL;
	static int it = 0;

	if (id == MA_CTRL_DEV_FIRST)
		it = 0;

	for (; it < IN_MAX_DEVS; it++) {
		name = in_get_dev_name(it, 1, 1);
		if (name != NULL)
			break;
	}

	it++;
	return name;
}

static int mh_saveloadcfg(int id, int keys);
static const char *mgn_saveloadcfg(int id, int *offs);

const char *indev0_names[] = { "none", "3 button pad", "6 button pad", "Team player", "4 way play", NULL };
const char *indev1_names[] = { "none", "3 button pad", "6 button pad", NULL };

static char h_play34[] = "only for MD(+add-ons) with Team/4 way";

static menu_entry e_menu_keyconfig[] =
{
	mee_handler_id("Player 1",          MA_CTRL_PLAYER1,    key_config_loop_wrap),
	mee_handler_id("Player 2",          MA_CTRL_PLAYER2,    key_config_loop_wrap),
	mee_handler_id_h("Player 3",        MA_CTRL_PLAYER3,    key_config_loop_wrap, h_play34),
	mee_handler_id_h("Player 4",        MA_CTRL_PLAYER4,    key_config_loop_wrap, h_play34),
	mee_handler_id("Emulator controls", MA_CTRL_EMU,        key_config_loop_wrap),
	mee_enum      ("Input device 1",    MA_OPT_INPUT_DEV0,  currentConfig.input_dev0, indev0_names),
	mee_enum      ("Input device 2",    MA_OPT_INPUT_DEV1,  currentConfig.input_dev1, indev1_names),
	mee_range     ("Turbo rate",        MA_CTRL_TURBO_RATE, currentConfig.turbo_rate, 1, 30),
	mee_range     ("Analog deadzone",   MA_CTRL_DEADZONE,   currentConfig.analog_deadzone, 1, 99),
	mee_cust_nosave("Save global config",       MA_OPT_SAVECFG, mh_saveloadcfg, mgn_saveloadcfg),
	mee_cust_nosave("Save cfg for loaded game", MA_OPT_SAVECFG_GAME, mh_saveloadcfg, mgn_saveloadcfg),
	mee_label     (""),
	mee_label     ("Input devices:"),
	mee_label_mk  (MA_CTRL_DEV_FIRST, mgn_dev_name),
	mee_label_mk  (MA_CTRL_DEV_NEXT,  mgn_dev_name),
	mee_label_mk  (MA_CTRL_DEV_NEXT,  mgn_dev_name),
	mee_label_mk  (MA_CTRL_DEV_NEXT,  mgn_dev_name),
	mee_label_mk  (MA_CTRL_DEV_NEXT,  mgn_dev_name),
	mee_label_mk  (MA_CTRL_DEV_NEXT,  mgn_dev_name),
	mee_label_mk  (MA_CTRL_DEV_NEXT,  mgn_dev_name),
	mee_end,
};

static int menu_loop_keyconfig(int id, int keys)
{
	static int sel = 0;

	me_enable(e_menu_keyconfig, MA_OPT_SAVECFG_GAME, PicoGameLoaded);
	me_loop_d(e_menu_keyconfig, &sel, menu_draw_prep, NULL);

	PicoSetInputDevice(0, currentConfig.input_dev0);
	PicoSetInputDevice(1, currentConfig.input_dev1);

	return 0;
}

// ------------ MD options menu ------------

static const char h_fmfilter[] = "improves sound quality but is noticeably slower\n"
				"best option if native rate isn't working";

static menu_entry e_menu_md_options[] =
{
	mee_enum      ("Renderer",        MA_OPT_RENDERER, currentConfig.renderer, renderer_names),
	mee_onoff_h   ("FM filtering",    MA_OPT_FM_FILTER, PicoIn.opt, POPT_EN_FM_FILTER, h_fmfilter),
	mee_end,
};

static int menu_loop_md_options(int id, int keys)
{
	static int sel = 0;

	me_enable(e_menu_md_options, MA_OPT_RENDERER, renderer_names[0] != NULL);
	me_loop_d(e_menu_md_options, &sel, menu_draw_prep, NULL);

	return 0;
}

// ------------ SCD options menu ------------

static const char h_cdleds[] = "Show power/CD LEDs of emulated console";
static const char h_cdda[]   = "Play audio tracks from mp3s/wavs/bins";
static const char h_cdpcm[]  = "Emulate PCM audio chip for effects/voices/music";
static const char h_srcart[] = "Emulate the save RAM cartridge accessory\n"
				"most games don't need this";
static const char h_scfx[]   = "Emulate scale/rotate ASIC chip for graphics effects\n"
				"disable to improve performance";

static menu_entry e_menu_cd_options[] =
{
	mee_onoff_h("CD LEDs",              MA_CDOPT_LEDS,          currentConfig.EmuOpt, EOPT_EN_CD_LEDS, h_cdleds),
	mee_onoff_h("CDDA audio",           MA_CDOPT_CDDA,          PicoIn.opt, POPT_EN_MCD_CDDA, h_cdda),
	mee_onoff_h("PCM audio",            MA_CDOPT_PCM,           PicoIn.opt, POPT_EN_MCD_PCM, h_cdpcm),
	mee_onoff_h("SaveRAM cart",         MA_CDOPT_SAVERAM,       PicoIn.opt, POPT_EN_MCD_RAMCART, h_srcart),
	mee_onoff_h("Scale/Rot. fx",        MA_CDOPT_SCALEROT_CHIP, PicoIn.opt, POPT_EN_MCD_GFX, h_scfx),
	mee_end,
};

static int menu_loop_cd_options(int id, int keys)
{
	static int sel = 0;
	me_loop_d(e_menu_cd_options, &sel, menu_draw_prep, NULL);
	return 0;
}

// ------------ 32X options menu ------------

#ifndef NO_32X

// convert from multiplier of VClk
static int mh_opt_sh2cycles(int id, int keys)
{
	int *khz = (id == MA_32XOPT_MSH2_CYCLES) ?
		&currentConfig.msh2_khz : &currentConfig.ssh2_khz;

	if (keys & (PBTN_LEFT|PBTN_RIGHT))
		*khz += (keys & PBTN_LEFT) ? -50 : 50;
	if (keys & (PBTN_L|PBTN_R))
		*khz += (keys & PBTN_L) ? -500 : 500;

	if (*khz < 1)
		*khz = 1;
	else if (*khz > 0x7fffffff / 1000)
		*khz = 0x7fffffff / 1000;

	return 0;
}

static const char *mgn_opt_sh2cycles(int id, int *offs)
{
	int khz = (id == MA_32XOPT_MSH2_CYCLES) ?
		currentConfig.msh2_khz : currentConfig.ssh2_khz;

	sprintf(static_buff, "%d", khz);
	return static_buff;
}

static const char h_32x_enable[] = "Enable emulation of the 32X addon";
static const char h_pwm[]        = "Disabling may improve performance, but break sound";
static const char h_pwmopt[]     = "Enabling may improve performance, but break sound";
static const char h_sh2cycles[]  = "Cycles/millisecond (similar to DOSBox)\n"
				   "lower values speed up emulation but break games\n"
				   "at least 11000 recommended for compatibility";

static menu_entry e_menu_32x_options[] =
{
	mee_onoff_h   ("32X enabled",       MA_32XOPT_ENABLE_32X,  PicoIn.opt, POPT_EN_32X, h_32x_enable),
	mee_enum      ("32X renderer",      MA_32XOPT_RENDERER,    currentConfig.renderer32x, renderer_names32x),
	mee_onoff_h   ("PWM sound",         MA_32XOPT_PWM,         PicoIn.opt, POPT_EN_PWM, h_pwm),
	mee_onoff_h   ("PWM IRQ optimization", MA_OPT2_PWM_IRQ_OPT, PicoIn.opt, POPT_PWM_IRQ_OPT, h_pwmopt),
	mee_cust_h    ("Master SH2 cycles", MA_32XOPT_MSH2_CYCLES, mh_opt_sh2cycles, mgn_opt_sh2cycles, h_sh2cycles),
	mee_cust_h    ("Slave SH2 cycles",  MA_32XOPT_SSH2_CYCLES, mh_opt_sh2cycles, mgn_opt_sh2cycles, h_sh2cycles),
	mee_end,
};

static int menu_loop_32x_options(int id, int keys)
{
	static int sel = 0;

	me_enable(e_menu_32x_options, MA_32XOPT_RENDERER, renderer_names32x[0] != NULL);
	me_loop_d(e_menu_32x_options, &sel, menu_draw_prep, NULL);

	Pico32xSetClocks(currentConfig.msh2_khz * 1000, currentConfig.msh2_khz * 1000);

	return 0;
}

#endif

// ------------ SMS options menu ------------

#ifndef NO_SMS

static const char *sms_hardwares[] = { "auto", "Game Gear", "Master System", "SG-1000", "SC-3000", NULL };
static const char *gg_ghosting_opts[] = { "OFF", "weak", "normal", NULL };
static const char *sms_mappers[] = { "auto", "Sega", "Codemasters", "Korea", "Korea MSX", "Korea X-in-1", "Korea 4-Pak", "Korea Janggun", "Korea Nemesis", "Taiwan 8K RAM", "Korea XOR", "Sega 32K RAM", NULL };
static const char h_smsfm[] = "FM sound is only supported by few games\nOther games may crash with FM enabled";
static const char h_ghost[] = "simulates the inertia of the GG LCD display";

static menu_entry e_menu_sms_options[] =
{
	mee_enum      ("System",            MA_SMSOPT_HARDWARE, PicoIn.hwSelect, sms_hardwares),
	mee_enum_h    ("Game Gear LCD ghosting", MA_SMSOPT_GHOSTING, currentConfig.ghosting, gg_ghosting_opts, h_ghost),
	mee_onoff_h   ("FM Sound Unit",     MA_OPT2_ENABLE_YM2413, PicoIn.opt, POPT_EN_YM2413, h_smsfm),
	mee_enum      ("Cartridge mapping", MA_SMSOPT_MAPPER, PicoIn.mapper, sms_mappers),
	mee_end,
};

static int menu_loop_sms_options(int id, int keys)
{
	static int sel = 0;

	me_loop_d(e_menu_sms_options, &sel, menu_draw_prep, NULL);

	return 0;
}

#endif

// ------------ adv options menu ------------

static const char h_ovrclk[] = "Will break some games, keep at 0";

static menu_entry e_menu_adv_options[] =
{
	mee_onoff     ("Disable sprite limit",     MA_OPT2_NO_SPRITE_LIM, PicoIn.opt, POPT_DIS_SPRITE_LIM),
	mee_range_h   ("Overclock M68k (%)",       MA_OPT2_OVERCLOCK_M68K,currentConfig.overclock_68k, 0, 1000, h_ovrclk),
	mee_onoff     ("Emulate Z80",              MA_OPT2_ENABLE_Z80,    PicoIn.opt, POPT_EN_Z80),
	mee_onoff     ("Emulate YM2612 (FM)",      MA_OPT2_ENABLE_YM2612, PicoIn.opt, POPT_EN_FM),
	mee_onoff     ("Disable YM2612 SSG-EG",    MA_OPT2_DISABLE_YM_SSG,PicoIn.opt, POPT_DIS_FM_SSGEG),
	mee_onoff     ("Enable YM2612 DAC noise",  MA_OPT2_ENABLE_YM_DAC, PicoIn.opt, POPT_EN_FM_DAC),
	mee_onoff     ("Emulate SN76496 (PSG)",    MA_OPT2_ENABLE_SN76496,PicoIn.opt, POPT_EN_PSG),
	mee_onoff     ("Emulate Game Gear LCD",    MA_OPT2_ENABLE_GGLCD  ,PicoIn.opt, POPT_EN_GG_LCD),
	mee_onoff     ("Disable idle loop patching",MA_OPT2_NO_IDLE_LOOPS,PicoIn.opt, POPT_DIS_IDLE_DET),
	mee_onoff     ("Disable frame limiter",    MA_OPT2_NO_FRAME_LIMIT,currentConfig.EmuOpt, EOPT_NO_FRMLIMIT),
	mee_onoff     ("Enable dynarecs",          MA_OPT2_DYNARECS,      PicoIn.opt, POPT_EN_DRC),
	MENU_OPTIONS_ADV
	mee_end,
};

static int menu_loop_adv_options(int id, int keys)
{
	static int sel = 0;

	me_loop_d(e_menu_adv_options, &sel, menu_draw_prep, NULL);
	PicoIn.overclockM68k = currentConfig.overclock_68k; // int vs short

	return 0;
}

// ------------ sound options menu ------------

static int sndrate_prevnext(int rate, int dir)
{
	static const int rates[] = { 8000, 11025, 16000, 22050, 44100, 53000 };
	int rate_count = sizeof(rates)/sizeof(rates[0]);
	int i;

	for (i = 0; i < rate_count; i++)
		if (rates[i] == rate) break;

	i += dir ? 1 : -1;
	if (i >= rate_count) {
		if (!(PicoIn.opt & POPT_EN_STEREO)) {
			PicoIn.opt |= POPT_EN_STEREO;
			return rates[0];
		}
		return rates[rate_count-1];
	}
	if (i < 0) {
		if (PicoIn.opt & POPT_EN_STEREO) {
			PicoIn.opt &= ~POPT_EN_STEREO;
			return rates[rate_count-1];
		}
		return rates[0];
	}
	return rates[i];
}

static int mh_opt_snd(int id, int keys)
{
	PicoIn.sndRate = sndrate_prevnext(PicoIn.sndRate, keys & PBTN_RIGHT);
	return 0;
}

static const char *mgn_opt_sound(int id, int *offs)
{
	const char *str2;
	*offs = -8;
	str2 = (PicoIn.opt & POPT_EN_STEREO) ? "stereo" : "mono";
	if (PicoIn.sndRate > 52000 && PicoIn.sndRate < 54000)
		sprintf(static_buff, "native  %s", str2);
	else	sprintf(static_buff, "%5iHz %s", PicoIn.sndRate, str2);
	return static_buff;
}

static int mh_opt_alpha(int id, int keys)
{
	int val = (PicoIn.sndFilterAlpha * 100 + 0x08000) / 0x10000;
	if (keys & PBTN_LEFT)	val--;
	if (keys & PBTN_RIGHT)	val++;
	if (val <  1)           val = 1;
	if (val > 99)           val = 99;
	PicoIn.sndFilterAlpha = val * 0x10000 / 100;
	return 0;
}

static const char *mgn_opt_alpha(int id, int *offs)
{
	int val = (PicoIn.sndFilterAlpha * 100 + 0x08000) / 0x10000;
	sprintf(static_buff, "0.%02d", val);
	return static_buff;
}

static const char h_quality[] = "native is the Megadrive sound chip rate (~53000),\n"
				"best quality, but might not work on some devices";
static const char h_lowpass[] = "Low pass filter for sound closer to real hardware";

static menu_entry e_menu_snd_options[] =
{
	mee_onoff     ("Enable sound",    MA_OPT_ENABLE_SOUND,  currentConfig.EmuOpt, EOPT_EN_SOUND),
	mee_cust_h    ("Sound Quality",   MA_OPT_SOUND_QUALITY, mh_opt_snd, mgn_opt_sound, h_quality),
	mee_onoff_h   ("Sound filter",    MA_OPT_SOUND_FILTER,  PicoIn.opt, POPT_EN_SNDFILTER, h_lowpass),
	mee_cust      ("Filter strength", MA_OPT_SOUND_ALPHA,   mh_opt_alpha, mgn_opt_alpha),
	mee_end,
};

static int menu_loop_snd_options(int id, int keys)
{
	static int sel = 0;

	if (PicoIn.sndRate > 52000 && PicoIn.sndRate < 54000)
		PicoIn.sndRate = 53000;
	me_loop_d(e_menu_snd_options, &sel, menu_draw_prep, NULL);

	return 0;
}

// ------------ gfx options menu ------------

static const char h_gamma[] = "Gamma/brightness adjustment (default 1.00)";

static const char *mgn_opt_fskip(int id, int *offs)
{
	if (currentConfig.Frameskip < 0)
		return "Auto";
	sprintf(static_buff, "%d", currentConfig.Frameskip);
	return static_buff;
}

static const char *mgn_aopt_gamma(int id, int *offs)
{
	sprintf(static_buff, "%i.%02i", currentConfig.gamma / 100, currentConfig.gamma % 100);
	return static_buff;
}

static menu_entry e_menu_gfx_options[] =
{
	mee_enum      ("Video output mode", MA_OPT_VOUT_MODE, plat_target.vout_method, men_dummy),
	mee_range_cust("Frameskip",         MA_OPT_FRAMESKIP, currentConfig.Frameskip, -1, 16, mgn_opt_fskip),
	mee_range     ("Max auto frameskip",MA_OPT2_MAX_FRAMESKIP, currentConfig.max_skip, 1, 10),
	mee_enum      ("Filter",            MA_OPT3_FILTERING, currentConfig.filter, men_dummy),
	mee_range_cust_h("Gamma correction",MA_OPT2_GAMMA, currentConfig.gamma, 1, 300, mgn_aopt_gamma, h_gamma),
	MENU_OPTIONS_GFX
	mee_end,
};

static int menu_loop_gfx_options(int id, int keys)
{
	static int sel = 0;

	me_loop_d(e_menu_gfx_options, &sel, menu_draw_prep, NULL);

	return 0;
}

// ------------ UI options menu ------------

static const char *men_confirm_save[] = { "OFF", "writes", "loads", "both", NULL };
static const char h_confirm_save[]    = "Ask for confirmation when overwriting save,\n"
					"loading state or both";

static menu_entry e_menu_ui_options[] =
{
	mee_onoff     ("Show FPS",                 MA_OPT_SHOW_FPS,       currentConfig.EmuOpt, EOPT_SHOW_FPS),
	mee_enum_h    ("Confirm savestate",        MA_OPT_CONFIRM_STATES, currentConfig.confirm_save, men_confirm_save, h_confirm_save),
	mee_onoff     ("Don't save last used ROM", MA_OPT2_NO_LAST_ROM,   currentConfig.EmuOpt, EOPT_NO_AUTOSVCFG),
	mee_end,
};

static int menu_loop_ui_options(int id, int keys)
{
	static int sel = 0;

	me_loop_d(e_menu_ui_options, &sel, menu_draw_prep, NULL);

	return 0;
}

// ------------ options menu ------------

static menu_entry e_menu_options[];

static void region_prevnext(int right)
{
	// jp_ntsc=1, jp_pal=2, usa=4, eu=8
	static const int rgn_orders[] = { 0x148, 0x184, 0x814, 0x418, 0x841, 0x481 };
	int i;

	if (right) {
		if (!PicoIn.regionOverride) {
			for (i = 0; i < 6; i++)
				if (rgn_orders[i] == PicoIn.autoRgnOrder) break;
			if (i < 5) PicoIn.autoRgnOrder = rgn_orders[i+1];
			else PicoIn.regionOverride=1;
		}
		else
			PicoIn.regionOverride <<= 1;
		if (PicoIn.regionOverride > 8)
			PicoIn.regionOverride = 8;
	} else {
		if (!PicoIn.regionOverride) {
			for (i = 0; i < 6; i++)
				if (rgn_orders[i] == PicoIn.autoRgnOrder) break;
			if (i > 0) PicoIn.autoRgnOrder = rgn_orders[i-1];
		}
		else
			PicoIn.regionOverride >>= 1;
	}
}

static int mh_opt_misc(int id, int keys)
{
	switch (id) {
	case MA_OPT_REGION:
		region_prevnext(keys & PBTN_RIGHT);
		break;
	default:
		break;
	}
	return 0;
}

static int mh_saveloadcfg(int id, int keys)
{
	int ret;

	if (keys & (PBTN_LEFT|PBTN_RIGHT)) { // multi choice
		config_slot += (keys & PBTN_LEFT) ? -1 : 1;
		if (config_slot < 0) config_slot = 9;
		else if (config_slot > 9) config_slot = 0;
		me_enable(e_menu_options, MA_OPT_LOADCFG, config_slot != config_slot_current);
		return 0;
	}

	switch (id) {
	case MA_OPT_SAVECFG:
	case MA_OPT_SAVECFG_GAME:
		if (emu_write_config(id == MA_OPT_SAVECFG_GAME ? 1 : 0))
			menu_update_msg("config saved");
		else
			menu_update_msg("failed to write config");
		break;
	case MA_OPT_LOADCFG:
		ret = emu_read_config(rom_fname_loaded, 1);
		if (!ret) ret = emu_read_config(NULL, 1);
		if (ret)  menu_update_msg("config loaded");
		else      menu_update_msg("failed to load config");
		break;
	default:
		return 0;
	}

	return 1;
}

static int mh_restore_defaults(int id, int keys)
{
	emu_set_defconfig();
	menu_update_msg("defaults restored");
	return 1;
}

static const char *mgn_opt_region(int id, int *offs)
{
	static const char *names[] = { "Auto", "      Japan NTSC", "      Japan PAL", "      USA", "      Europe" };
	static const char *names_short[] = { "", " JP", " JP", " US", " EU" };
	int code = PicoIn.regionOverride;
	int u, i = 0;

	*offs = -6;
	if (code) {
		code <<= 1;
		while ((code >>= 1)) i++;
		if (i > 4)
			return "unknown";
		return names[i];
	} else {
		strcpy(static_buff, "Auto:");
		for (u = 0; u < 3; u++) {
			code = (PicoIn.autoRgnOrder >> u*4) & 0xf;
			for (i = 0; code; code >>= 1, i++)
				;
			strcat(static_buff, names_short[i]);
		}
		return static_buff;
	}
}

static const char *mgn_saveloadcfg(int id, int *offs)
{
	static_buff[0] = 0;
	if (config_slot != 0)
		sprintf(static_buff, "[%i]", config_slot);
	return static_buff;
}

static menu_entry e_menu_options[] =
{
	mee_range     ("Save slot",                MA_OPT_SAVE_SLOT,     state_slot, 0, 9),
	mee_cust      ("Region",                   MA_OPT_REGION,        mh_opt_misc, mgn_opt_region),
	mee_range     ("",                         MA_OPT_CPU_CLOCKS,    currentConfig.CPUclock, 20, 3200),
	mee_handler   ("[Interface options]",      menu_loop_ui_options),
	mee_handler   ("[Display options]",        menu_loop_gfx_options),
	mee_handler   ("[Sound options]",          menu_loop_snd_options),
	mee_handler   ("[MD/Genesis options]",     menu_loop_md_options),
	mee_handler   ("  [Sega/Mega CD add-on]",  menu_loop_cd_options),
#ifndef NO_32X
	mee_handler   ("  [32X add-on]",           menu_loop_32x_options),
#endif
#ifndef NO_SMS
	mee_handler   ("[SG/SMS/GG options]",      menu_loop_sms_options),
#endif
	mee_handler   ("[Advanced options]",       menu_loop_adv_options),
	mee_cust_nosave("Save global config",      MA_OPT_SAVECFG, mh_saveloadcfg, mgn_saveloadcfg),
	mee_cust_nosave("Save cfg for loaded game",MA_OPT_SAVECFG_GAME, mh_saveloadcfg, mgn_saveloadcfg),
	mee_cust_nosave("Load cfg from profile",   MA_OPT_LOADCFG, mh_saveloadcfg, mgn_saveloadcfg),
	mee_handler   ("Restore defaults",         mh_restore_defaults),
	mee_end,
};

static int menu_loop_options(int id, int keys)
{
	static int sel = 0;

	me_enable(e_menu_options, MA_OPT_SAVECFG_GAME, PicoGameLoaded);
	me_enable(e_menu_options, MA_OPT_LOADCFG, config_slot != config_slot_current);

	me_loop_d(e_menu_options, &sel, menu_draw_prep, NULL);

	return 0;
}

// ------------ debug menu ------------

#include <pico/debug.h>

extern void SekStepM68k(void);

static void mplayer_loop(void)
{
	pemu_sound_start();

	while (1)
	{
		PDebugZ80Frame();
		if (in_menu_wait_any(NULL, 0) & PBTN_MA3)
			break;
		emu_sound_wait();
	}

	emu_sound_stop();
}

static void draw_text_debug(const char *str, int skip, int from)
{
	const char *p;
	int line;

	p = str;
	while (skip-- > 0)
	{
		while (*p && *p != '\n')
			p++;
		if (*p == 0 || p[1] == 0)
			return;
		p++;
	}

	str = p;
	for (line = from; line < g_menuscreen_h / me_sfont_h; line++)
	{
		smalltext_out16(1, line * me_sfont_h, str, 0xffff);
		while (*p && *p != '\n')
			p++;
		if (*p == 0)
			break;
		p++; str = p;
	}
}

#ifdef __GNUC__
#define COMPILER "gcc " __VERSION__
#else
#define COMPILER
#endif

static void draw_frame_debug(void)
{
	char layer_str[48] = "layers:                   ";
	struct PicoVideo *pv = &Pico.video;

	if (!(pv->debug_p & PVD_KILL_B))    memcpy(layer_str +  8, "B", 1);
	if (!(pv->debug_p & PVD_KILL_A))    memcpy(layer_str + 10, "A", 1);
	if (!(pv->debug_p & PVD_KILL_S_LO)) memcpy(layer_str + 12, "spr_lo", 6);
	if (!(pv->debug_p & PVD_KILL_S_HI)) memcpy(layer_str + 19, "spr_hi", 6);
	if (!(pv->debug_p & PVD_KILL_32X))  memcpy(layer_str + 26, "32x", 4);

	pemu_forced_frame(1, 0);
	make_bg(1, 1);

	smalltext_out16(4, 1, "build: r" REVISION "  "__DATE__ " " __TIME__ " " COMPILER, 0xffff);
	smalltext_out16(4, g_menuscreen_h - me_sfont_h, layer_str, 0xffff);
}

static void debug_menu_loop(void)
{
	struct PicoVideo *pv = &Pico.video;
	int inp, mode = 0;
	int spr_offs = 0, dumped = 0;
	char *tmp;

	while (1)
	{
		menu_draw_begin(1, 0);
		g_screen_ptr = g_menuscreen_ptr;
		g_screen_width = g_menuscreen_w;
		g_screen_height = g_menuscreen_h;
		g_screen_ppitch = g_menuscreen_pp;
		switch (mode)
		{
			case 0: tmp = PDebugMain();
				plat_debug_cat(tmp);
				draw_text_debug(tmp, 0, 0);
				if (dumped) {
					smalltext_out16(g_menuscreen_w - 6 * me_sfont_h,
						g_menuscreen_h - me_mfont_h, "dumped", 0xffff);
					dumped = 0;
				}
				break;
			case 1: draw_frame_debug();
				break;
			case 2: pemu_forced_frame(1, 0);
				make_bg(1, 1);
				PDebugShowSpriteStats((unsigned short *)g_menuscreen_ptr
					+ (g_menuscreen_h/2 - 240/2) * g_menuscreen_pp
					+ g_menuscreen_w/2 - 320/2, g_menuscreen_pp);
				break;
			case 3: menuscreen_memset_lines(g_menuscreen_ptr, 0, g_menuscreen_h);
				PDebugShowPalette(g_menuscreen_ptr, g_menuscreen_pp);
				PDebugShowSprite((unsigned short *)g_menuscreen_ptr
					+ g_menuscreen_pp * 120 + g_menuscreen_w / 2 + 16,
					g_menuscreen_pp, spr_offs);
				draw_text_debug(PDebugSpriteList(), spr_offs, 6);
				break;
			case 4: tmp = PDebug32x();
				draw_text_debug(tmp, 0, 0);
				break;
		}
		menu_draw_end();

		inp = in_menu_wait(PBTN_MOK|PBTN_MBACK|PBTN_MA2|PBTN_MA3|PBTN_L|PBTN_R |
					PBTN_UP|PBTN_DOWN|PBTN_LEFT|PBTN_RIGHT, NULL, 70);
		if (inp & PBTN_MBACK) return;
		if (inp & PBTN_L) { mode--; if (mode < 0) mode = 4; }
		if (inp & PBTN_R) { mode++; if (mode > 4) mode = 0; }
		switch (mode)
		{
			case 0:
				if (inp & PBTN_MOK)
					PDebugCPUStep();
				if (inp & PBTN_MA3) {
					while (inp & PBTN_MA3)
						inp = in_menu_wait_any(NULL, -1);
					mplayer_loop();
				}
				if ((inp & (PBTN_MA2|PBTN_LEFT)) == (PBTN_MA2|PBTN_LEFT)) {
					mkdir("dumps", 0777);
					PDebugDumpMem();
					while (inp & PBTN_MA2) inp = in_menu_wait_any(NULL, -1);
					dumped = 1;
				}
				break;
			case 1:
				if (inp & PBTN_LEFT)  pv->debug_p ^= PVD_KILL_B;
				if (inp & PBTN_RIGHT) pv->debug_p ^= PVD_KILL_A;
				if (inp & PBTN_DOWN)  pv->debug_p ^= PVD_KILL_S_LO;
				if (inp & PBTN_UP)    pv->debug_p ^= PVD_KILL_S_HI;
				if (inp & PBTN_MA2)   pv->debug_p ^= PVD_KILL_32X;
				if (inp & PBTN_MOK) {
					PicoIn.sndOut = NULL; // just in case
					PicoIn.skipFrame = 1;
					PicoFrame();
					PicoIn.skipFrame = 0;
					while (inp & PBTN_MOK) inp = in_menu_wait_any(NULL, -1);
				}
				break;
			case 3:
				if (inp & PBTN_DOWN)  spr_offs++;
				if (inp & PBTN_UP)    spr_offs--;
				if (spr_offs < 0) spr_offs = 0;
				break;
		}
	}
}

// ------------ main menu ------------

static void draw_frame_credits(void)
{
	smalltext_out16(4, 1, "build: " __DATE__ " " __TIME__, 0xe7fc);
}

static const char credits[] =
	"PicoDrive v" VERSION "\n"
	"(c) notaz, 2006-2013; irixxxx, 2018-2021\n\n"
	"Credits:\n"
	"fDave: initial code\n"
#ifdef EMU_C68K
	"      Cyclone 68000 core\n"
#else
	"Stef, Chui: FAME/C 68k core\n"
#endif
#ifdef _USE_DRZ80
	"Reesy & FluBBa: DrZ80 core\n"
#else
	"Stef, NJ: CZ80 core\n"
#endif
	"MAME devs: SH2, YM2612 and SN76496 cores\n"
	"Eke, Stef: some Sega CD code\n"
	"Inder, ketchupgun: graphics\n"
#ifdef __GP2X__
	"Squidge: mmuhack\n"
	"Dzz: ARM940 sample\n"
#endif
	"\n"
	"special thanks (for docs, ideas):\n"
	" Charles MacDonald, Haze,\n"
	" Stephane Dallongeville,\n"
	" Lordus, Exophase, Rokas,\n"
	" Eke, Nemesis, Tasco Deluxe";

static void menu_main_draw_status(void)
{
	static time_t last_bat_read = 0;
	static int last_bat_val = -1;
	unsigned short *bp = g_menuscreen_ptr;
	int bat_h = me_mfont_h * 2 / 3;
	int i, u, w, wfill, batt_val;
	struct tm *tmp;
	time_t ltime;
	char time_s[16];

	if (!(currentConfig.EmuOpt & EOPT_SHOW_RTC))
		return;

	ltime = time(NULL);
	tmp = gmtime(&ltime);
	strftime(time_s, sizeof(time_s), "%H:%M", tmp);

	text_out16(g_menuscreen_w - me_mfont_w * 6, me_mfont_h + 2, time_s);

	if (ltime - last_bat_read > 10) {
		last_bat_read = ltime;
		last_bat_val = batt_val = plat_target_bat_capacity_get();
	}
	else
		batt_val = last_bat_val;

	if (batt_val < 0 || batt_val > 100)
		return;

	/* battery info */
	bp += (me_mfont_h * 2 + 2) * g_menuscreen_pp + g_menuscreen_w - me_mfont_w * 3 - 3;
	for (i = 0; i < me_mfont_w * 2; i++)
		bp[i] = menu_text_color;
	for (i = 0; i < me_mfont_w * 2; i++)
		bp[i + g_menuscreen_pp * bat_h] = menu_text_color;
	for (i = 0; i <= bat_h; i++)
		bp[i * g_menuscreen_pp] =
		bp[i * g_menuscreen_pp + me_mfont_w * 2] = menu_text_color;
	for (i = 2; i < bat_h - 1; i++)
		bp[i * g_menuscreen_pp - 1] =
		bp[i * g_menuscreen_pp - 2] = menu_text_color;

	w = me_mfont_w * 2 - 1;
	wfill = batt_val * w / 100;
	for (u = 1; u < bat_h; u++)
		for (i = 0; i < wfill; i++)
			bp[(w - i) + g_menuscreen_pp * u] = menu_text_color;
}

static int main_menu_handler(int id, int keys)
{
	const char *ret_name;

	switch (id)
	{
	case MA_MAIN_RESUME_GAME:
		if (PicoGameLoaded)
			return 1;
		break;
	case MA_MAIN_SAVE_STATE:
		if (PicoGameLoaded)
			return menu_loop_savestate(0);
		break;
	case MA_MAIN_LOAD_STATE:
		if (PicoGameLoaded)
			return menu_loop_savestate(1);
		break;
	case MA_MAIN_RESET_GAME:
		if (PicoGameLoaded) {
			emu_reset_game();
			return 1;
		}
		break;
	case MA_MAIN_LOAD_ROM:
		rom_fname_reload = NULL;
		ret_name = menu_loop_romsel(rom_fname_loaded,
			sizeof(rom_fname_loaded), rom_exts, NULL);
		if (ret_name != NULL) {
			lprintf("selected file: %s\n", ret_name);
			rom_fname_reload = ret_name;
			engineState = PGS_ReloadRom;
			return 1;
		}
		break;
	case MA_MAIN_CHANGE_CD:
		if (PicoIn.AHW & PAHW_MCD) {
			// if cd is loaded, cdd_unload() triggers eject and
			// returns 1, else we'll select and load new CD here
			if (!cdd_unload())
				menu_loop_tray();
			return 1;
		}
		break;
	case MA_MAIN_CREDITS:
		draw_menu_message(credits, draw_frame_credits);
		in_menu_wait(PBTN_MOK|PBTN_MBACK, NULL, 70);
		break;
	case MA_MAIN_EXIT:
		engineState = PGS_Quit;
		return 1;
	case MA_MAIN_PATCHES:
		if (PicoGameLoaded && PicoPatches) {
			menu_loop_patches();
			PicoPatchApply();
			menu_update_msg("Patches applied");
		}
		break;
	default:
		lprintf("%s: something unknown selected\n", __FUNCTION__);
		break;
	}

	return 0;
}

static menu_entry e_menu_main[] =
{
	mee_label     ("PicoDrive " VERSION),
	mee_label     (""),
	mee_label     (""),
	mee_label     (""),
	mee_handler_id("Resume game",        MA_MAIN_RESUME_GAME, main_menu_handler),
	mee_handler_id("Save State",         MA_MAIN_SAVE_STATE,  main_menu_handler),
	mee_handler_id("Load State",         MA_MAIN_LOAD_STATE,  main_menu_handler),
	mee_handler_id("Reset game",         MA_MAIN_RESET_GAME,  main_menu_handler),
	mee_handler_id("Load new ROM/ISO",   MA_MAIN_LOAD_ROM,    main_menu_handler),
	mee_handler_id("Change CD/ISO",      MA_MAIN_CHANGE_CD,   main_menu_handler),
	mee_handler   ("Change options",                          menu_loop_options),
	mee_handler   ("Configure controls",                      menu_loop_keyconfig),
	mee_handler_id("Credits",            MA_MAIN_CREDITS,     main_menu_handler),
	mee_handler_id("Patches / GameGenie",MA_MAIN_PATCHES,     main_menu_handler),
	mee_handler_id("Exit",               MA_MAIN_EXIT,        main_menu_handler),
	mee_end,
};

void menu_loop(void)
{
	static int sel = 0;

	me_enable(e_menu_main, MA_MAIN_RESUME_GAME, PicoGameLoaded);
	me_enable(e_menu_main, MA_MAIN_SAVE_STATE,  PicoGameLoaded);
	me_enable(e_menu_main, MA_MAIN_LOAD_STATE,  PicoGameLoaded);
	me_enable(e_menu_main, MA_MAIN_RESET_GAME,  PicoGameLoaded);
	me_enable(e_menu_main, MA_MAIN_CHANGE_CD,   PicoIn.AHW & PAHW_MCD);
	me_enable(e_menu_main, MA_MAIN_PATCHES, PicoPatches != NULL);

	menu_enter(PicoGameLoaded);
	in_set_config_int(0, IN_CFG_BLOCKING, 1);
	me_loop_d(e_menu_main, &sel, menu_draw_prep, menu_main_draw_status);

	if (PicoGameLoaded) {
		if (engineState == PGS_Menu)
			engineState = PGS_Running;
		/* wait until menu, ok, back is released */
		while (in_menu_wait_any(NULL, 50) & (PBTN_MENU|PBTN_MOK|PBTN_MBACK))
			;
	}

	in_set_config_int(0, IN_CFG_BLOCKING, 0);
	plat_video_menu_leave();
}

// --------- CD tray close menu ----------

static int mh_tray_load_cd(int id, int keys)
{
	const char *ret_name;

	rom_fname_reload = NULL;
	ret_name = menu_loop_romsel(rom_fname_loaded,
			sizeof(rom_fname_loaded), rom_exts, NULL);
	if (ret_name == NULL)
		return 0;

	rom_fname_reload = ret_name;
	engineState = PGS_RestartRun;
	return emu_swap_cd(ret_name);
}

static int mh_tray_nothing(int id, int keys)
{
	return 1;
}

static menu_entry e_menu_tray[] =
{
	mee_label  ("The CD tray has opened."),
	mee_label  (""),
	mee_label  (""),
	mee_handler("Load CD image",  mh_tray_load_cd),
	mee_handler("Insert nothing", mh_tray_nothing),
	mee_end,
};

int menu_loop_tray(void)
{
	int ret = 1, sel = 0;

	menu_enter(PicoGameLoaded);

	in_set_config_int(0, IN_CFG_BLOCKING, 1);
	me_loop_d(e_menu_tray, &sel, menu_draw_prep, NULL);

	if (engineState != PGS_RestartRun) {
		engineState = PGS_RestartRun;
		ret = 0; /* no CD inserted */
	}

	while (in_menu_wait_any(NULL, 50) & (PBTN_MENU|PBTN_MOK|PBTN_MBACK))
		;
	in_set_config_int(0, IN_CFG_BLOCKING, 0);
	plat_video_menu_leave();

	return ret;
}

void menu_update_msg(const char *msg)
{
	strncpy(menu_error_msg, msg, sizeof(menu_error_msg));
	menu_error_msg[sizeof(menu_error_msg) - 1] = 0;

	menu_error_time = plat_get_ticks_ms();
	lprintf("msg: %s\n", menu_error_msg);
}

// ------------ util ------------

/* hidden options for config engine only */
static menu_entry e_menu_hidden[] =
{
	mee_onoff("Accurate sprites", MA_OPT_ACC_SPRITES, PicoIn.opt, POPT_ACC_SPRITES),
	mee_onoff("autoload savestates", MA_OPT_AUTOLOAD_SAVE, g_autostateld_opt, 1),
	mee_onoff("SDL fullscreen mode", MA_OPT_VOUT_FULL, plat_target.vout_fullscreen, 1),
	mee_end,
};

static menu_entry *e_menu_table[] =
{
	e_menu_options,
	e_menu_ui_options,
	e_menu_snd_options,
	e_menu_gfx_options,
	e_menu_adv_options,
	e_menu_md_options,
	e_menu_cd_options,
#ifndef NO_32X
	e_menu_32x_options,
#endif
#ifndef NO_SMS
	e_menu_sms_options,
#endif
	e_menu_keyconfig,
	e_menu_hidden,
};

static menu_entry *me_list_table = NULL;
static menu_entry *me_list_i = NULL;

menu_entry *me_list_get_first(void)
{
	me_list_table = me_list_i = e_menu_table[0];
	return me_list_i;
}

menu_entry *me_list_get_next(void)
{
	int i;

	me_list_i++;
	if (me_list_i->name != NULL)
		return me_list_i;

	for (i = 0; i < array_size(e_menu_table); i++)
		if (me_list_table == e_menu_table[i])
			break;

	if (i + 1 < array_size(e_menu_table))
		me_list_table = me_list_i = e_menu_table[i + 1];
	else
		me_list_table = me_list_i = NULL;

	return me_list_i;
}

void menu_init(void)
{
	int i;

	menu_init_base();

	i = 0;
#if defined(_SVP_DRC) || defined(DRC_SH2)
	i = 1;
#endif
	me_enable(e_menu_adv_options, MA_OPT2_DYNARECS, i);

	i = me_id2offset(e_menu_gfx_options, MA_OPT_VOUT_MODE);
	e_menu_gfx_options[i].data = plat_target.vout_methods;
	me_enable(e_menu_gfx_options, MA_OPT_VOUT_MODE,
		plat_target.vout_methods != NULL);

	i = me_id2offset(e_menu_gfx_options, MA_OPT3_FILTERING);
	e_menu_gfx_options[i].data = plat_target.hwfilters;
	me_enable(e_menu_gfx_options, MA_OPT3_FILTERING,
		plat_target.hwfilters != NULL);

	me_enable(e_menu_gfx_options, MA_OPT2_GAMMA,
                plat_target.gamma_set != NULL);

	i = me_id2offset(e_menu_options, MA_OPT_CPU_CLOCKS);
	e_menu_options[i].enabled = 0;
	if (plat_target.cpu_clock_set != NULL) {
		e_menu_options[i].name = "CPU clock";
		e_menu_options[i].enabled = 1;
	}
}
