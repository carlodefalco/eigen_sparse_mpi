#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([eigen_sparse_mpi], [0.1], [carlo.defalco@polimi.it])
AC_CONFIG_SRCDIR([src/distributed_sparse_matrix.h])
AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([subdir-objects])

# Init libtool.
LT_PREREQ([2.2.2])
LT_INIT([disable-static dlopen])
AC_SUBST([LIBTOOL_DEPS])

# Checks for programs.
AC_PROG_CXX
AC_LANG_PUSH([C++])

# Checks for libraries.

# Checks for header files.

AC_ARG_WITH([eigen3-home],
[AS_HELP_STRING([--with-eigen3-home=DIR],
[Select the directory where eigen3 is installed.])],
[EIGEN3_HOME=$withval],
[EIGEN3_HOME=/opt/local/include/eigen3])

EIGEN3_CPPFLAGS="-I$EIGEN3_HOME"
save_CPPFLAGS=$CPPFLAGS
CPPFLAGS="$EIGEN3_CPPFLAGS $CPPFLAGS"

AC_CHECK_HEADER(Eigen/Sparse,
                AC_DEFINE([HAVE_EIGEN3],[1],[Define if you have lis library])
                AC_SUBST(EIGEN3_CPPFLAGS, "$EIGEN3_CPPFLAGS"),
                AC_MSG_ERROR([Eigen/Sparse not found. I cowardly refuse to continue.]))

CPPFLAGS=$save_CPPFLAGS

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.
AC_LANG_POP([C++])
AC_CONFIG_FILES([Makefile
                src/Makefile
                test/Makefile])

AC_OUTPUT
