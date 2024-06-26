#ifndef LIBPICOFE_POSIX_H
#define LIBPICOFE_POSIX_H

/* define POSIX stuff: dirent, scandir, getcwd, mkdir */
#if defined(__MACH__) || defined(__linux__) || defined(__MINGW32__)

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __MINGW32__
#warning hacks!
#define mkdir(pathname,mode) mkdir(pathname)
#define d_type d_ino
#define DT_REG 0
#define DT_DIR 0
#endif

#elif defined(__PSP__)

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

#ifndef DT_DIR
/* map PSP names to posix. needs special scandir() function to mask rwx bits */
#define d_type d_stat.st_attr
#define DT_LNK FIO_SO_IFLNK
#define DT_DIR FIO_SO_IFDIR
#define DT_REG FIO_SO_IFREG
#define DT_UNKNOWN 0
#endif

#else

#error "must provide posix"

#endif

#endif // LIBPICOFE_POSIX_H
