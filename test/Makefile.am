BIMIOLIB    = ../src/libeigen_sparse_mpi.la
AM_CPPFLAGS = -I$(top_srcdir)/src -I$(top_srcdir)/test -I$(top_builddir)/src @EIGEN3_CPPFLAGS@
AUTOMAKE_OPTIONS = nostdinc

CLEANFILES = 

check_PROGRAMS = sparse_test 

sparse_test_SOURCES =  sparse_test.cpp
sparse_test_CPPFLAGS =  $(AM_CPPFLAGS) 
CLEANFILES += 

TEST_EXTENSIONS = .m .sh .mpi
M_LOG_COMPILER = octave
SH_LOG_COMPILER = bash
MPI_LOG_COMPILER = mpiexec
AM_M_LOG_FLAGS = -q --no-gui --path $(top_srcdir)/script/m:$(top_srcdir)/data
AM_MPI_LOG_FLAGS = -np 2 

TESTS = sparse_test 
