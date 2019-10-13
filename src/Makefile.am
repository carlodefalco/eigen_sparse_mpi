AM_CPPFLAGS = -I$(top_srcdir)/src -I$(top_builddir)/src 
AUTOMAKE_OPTIONS = nostdinc

lib_LTLIBRARIES = libeigen_sparse_mpi.la

libeigen_sparse_mpi_la_SOURCES  = distributed_sparse_matrix.cpp
libeigen_sparse_mpi_la_CPPFLAGS = @OCTAVE_COMPILE_FLAGS@ $(AM_CPPFLAGS)

