# Configure paths for libFLAC
# "Inspired" by ogg.m4

dnl AM_PATH_LIBFLAC([ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl Test for libFLAC, and define LIBFLAC_CFLAGS, LIBFLAC_LIBS, LIBFLAC_LIBDIR
dnl
AC_DEFUN([AM_PATH_LIBFLAC],
[dnl
dnl Get the cflags and libraries
dnl
AC_ARG_WITH(libFLAC,[  --with-libFLAC=PFX   Prefix where libFLAC is installed (optional)], libFLAC_prefix="$withval", libFLAC_prefix="")
AC_ARG_WITH(libFLAC-libraries,[  --with-libFLAC-libraries=DIR   Directory where libFLAC library is installed (optional)], libFLAC_libraries="$withval", libFLAC_libraries="")
AC_ARG_WITH(libFLAC-includes,[  --with-libFLAC-includes=DIR   Directory where libFLAC header files are installed (optional)], libFLAC_includes="$withval", libFLAC_includes="")
AC_ARG_ENABLE(libFLACtest, [  --disable-libFLACtest       Do not try to compile and run a test libFLAC program],, enable_libFLACtest=yes)

  if test "x$libFLAC_libraries" != "x" ; then
    LIBFLAC_LIBS="-L$libFLAC_libraries"
  elif test "x$libFLAC_prefix" = "xno" || test "x$libFLAC_prefix" = "xyes" ; then
    LIBFLAC_LIBS=""
  elif test "x$libFLAC_prefix" != "x" ; then
    LIBFLAC_LIBS="-L$libFLAC_prefix/lib"
  elif test "x$prefix" != "xNONE"; then
    LIBFLAC_LIBS="-L$prefix/lib"
  fi

  if test "x$libFLAC_prefix" != "xno" ; then
    LIBFLAC_LIBS="$LIBFLAC_LIBS -lFLAC $OGG_LIBS -lm"
  fi

  if test "x$libFLAC_includes" != "x" ; then
    LIBFLAC_CFLAGS="-I$libFLAC_includes"
  elif test "x$libFLAC_prefix" != "x" ; then
    LIBFLAC_CFLAGS="-I$libFLAC_prefix/include"
  elif test "$prefix" != "xNONE"; then
    LIBFLAC_CFLAGS=""
  fi

  AC_MSG_CHECKING(for libFLAC)
  no_libFLAC=""


  if test "x$enable_libFLACtest" = "xyes" ; then
    ac_save_CFLAGS="$CFLAGS"
    ac_save_CXXFLAGS="$CXXFLAGS"
    ac_save_LIBS="$LIBS"
    ac_save_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    CFLAGS="$CFLAGS $LIBFLAC_CFLAGS"
    CXXFLAGS="$CXXFLAGS $LIBFLAC_CFLAGS"
    LIBS="$LIBS $LIBFLAC_LIBS"
    LD_LIBRARY_PATH="$LIBFLAC_LIBDIR:$LD_LIBRARY_PATH"
dnl
dnl Now check if the installed libFLAC is sufficiently new.
dnl
      rm -f conf.libFLACtest
      AC_TRY_RUN([
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <FLAC/format.h>

int main ()
{
  system("touch conf.libFLACtest");
  return 0;
}

],, no_libFLAC=yes,[echo $ac_n "cross compiling; assumed OK... $ac_c"])
       CFLAGS="$ac_save_CFLAGS"
       CXXFLAGS="$ac_save_CXXFLAGS"
       LIBS="$ac_save_LIBS"
       LD_LIBRARY_PATH="$ac_save_LD_LIBRARY_PATH"
  fi

  if test "x$no_libFLAC" = "x" ; then
     AC_MSG_RESULT(yes)
     ifelse([$1], , :, [$1])
  else
     AC_MSG_RESULT(no)
     if test -f conf.libFLACtest ; then
       :
     else
       echo "*** Could not run libFLAC test program, checking why..."
       CFLAGS="$CFLAGS $LIBFLAC_CFLAGS"
       CXXFLAGS="$CXXFLAGS $LIBFLAC_CFLAGS"
       LIBS="$LIBS $LIBFLAC_LIBS"
       LD_LIBRARY_PATH="$LIBFLAC_LIBDIR:$LD_LIBRARY_PATH"
       AC_TRY_LINK([
#include <stdio.h>
#include <FLAC/format.h>
],     [ return 0; ],
       [ echo "*** The test program compiled, but did not run. This usually means"
       echo "*** that the run-time linker is not finding libFLAC or finding the wrong"
       echo "*** version of libFLAC. If it is not finding libFLAC, you'll need to set your"
       echo "*** LD_LIBRARY_PATH environment variable, or edit /etc/ld.so.conf to point"
       echo "*** to the installed location  Also, make sure you have run ldconfig if that"
       echo "*** is required on your system"
       echo "***"
       echo "*** If you have an old version installed, it is best to remove it, although"
       echo "*** you may also be able to get things to work by modifying LD_LIBRARY_PATH"],
       [ echo "*** The test program failed to compile or link. See the file config.log for the"
       echo "*** exact error that occured. This usually means libFLAC was incorrectly installed"
       echo "*** or that you have moved libFLAC since it was installed. In the latter case, you"
       echo "*** may want to edit the libFLAC-config script: $LIBFLAC_CONFIG" ])
       CFLAGS="$ac_save_CFLAGS"
       CXXFLAGS="$ac_save_CXXFLAGS"
       LIBS="$ac_save_LIBS"
       LD_LIBRARY_PATH="$ac_save_LD_LIBRARY_PATH"
     fi
     LIBFLAC_CFLAGS=""
     LIBFLAC_LIBDIR=""
     LIBFLAC_LIBS=""
     ifelse([$2], , :, [$2])
  fi
  AC_SUBST(LIBFLAC_CFLAGS)
  AC_SUBST(LIBFLAC_LIBDIR)
  AC_SUBST(LIBFLAC_LIBS)
  rm -f conf.libFLACtest
])
