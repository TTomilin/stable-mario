This is libzip, a C library for reading, creating, and modifying zip
archives. Files can be added from data buffers, files, or compressed
data copied directly from other zip archives. Changes made without
closing the archive can be reverted. Decryption and encryption of
Winzip AES and decryption of legacy PKware encrypted files is
supported. The API is documented by man pages.

For more information, take a look at the included man pages.  You can
start with [libzip(3)](https://libzip.org/documentation/libzip.html), which lists
all others.  Example source code is in the `examples/` and  `src/` subdirectories.

If you have developed an application using libzip, you can find out
about API changes and how to adapt your code for them in the included
file [API-CHANGES.md](API-CHANGES.md).

See the [INSTALL.md](INSTALL.md) file for installation instructions and
dependencies.

If you make a binary distribution, please include a pointer to the
distribution site:
>	https://libzip.org/

The latest version can always be found there.  The official repository
is at [github](https://github.com/nih-at/libzip/).

There is a mailing list for developers using libzip.  You can
subscribe to it by sending a mail with the subject "subscribe
libzip-discuss" to minimalist at nih.at. List mail should be sent
to libzip-discuss at nih.at. Use this for bug reports or questions.

If you want to reach the authors in private, use <libzip@nih.at>.

![Travis Build Status](https://api.travis-ci.org/nih-at/libzip.svg?branch=master)
![Coverity Status](https://scan.coverity.com/projects/127/badge.svg?flat=1)
