AM_CPPFLAGS = -I$(top_srcdir)/src -I$(top_builddir)/src @EIGEN3_CPPFLAGS@
AUTOMAKE_OPTIONS = nostdinc

lib_LTLIBRARIES = libeigen_sparse_mpi.la

libeigen_sparse_mpi_la_SOURCES  = distributed_sparse_matrix.cpp mpi_alltoallv2.cpp
libeigen_sparse_mpi_la_CPPFLAGS = $(AM_CPPFLAGS)

include_HEADERS = distributed_sparse_matrix.h mpi_alltoallv2.h
