#ifndef __MDFN_SS_VDP2_COMMON_H
#define __MDFN_SS_VDP2_COMMON_H

enum
{
 RDBS_UNUSED = 0x0,
 RDBS_COEFF  = 0x1,
 RDBS_NAME   = 0x2,
 RDBS_CHAR   = 0x3
};

enum
{
 VCP_NBG0_NT = 0x0,
 VCP_NBG1_NT = 0x1,
 VCP_NBG2_NT = 0x2,
 VCP_NBG3_NT = 0x3,

 VCP_NBG0_CG = 0x4,
 VCP_NBG1_CG = 0x5,
 VCP_NBG2_CG = 0x6,
 VCP_NBG3_CG = 0x7,

 VCP_NBG0_VCS = 0xC,
 VCP_NBG1_VCS = 0xD,
 VCP_CPU = 0xE,
 VCP_NOP = 0xF
};

#endif
