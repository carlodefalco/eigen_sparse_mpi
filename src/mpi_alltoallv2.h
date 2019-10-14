/*
  Copyright (C) 2019 Carlo de Falco
  This software is distributed under the terms
  the terms of the GNU/GPL licence v3
*/


#ifndef HAVE_MPI_ALLTOALLV2_H
#define HAVE_MPI_ALLTOALLV2_H

#include <mpi.h>

/// This is an implementation of a vector all-to-all collective
/// communication directive that uses internally a set of non-blocking
/// MPI_Isend / MPI_Irecv calls. This is understood to be more efficient
/// than a standard MPI_alltoallv call when only a small subset of the
/// size^2 connections are actually to be established.
/// See for example this SO thread for motivation :
/// https://stackoverflow.com/questions/13505799/mpi-alltoallv-or-better-individual-send-and-recv-performance
int
MPI_Alltoallv2 (void ** sendbuf, int *sendcnts, MPI_Datatype sendtype,
                void ** recvbuf, int *recvcnts, MPI_Datatype recvtype,
                MPI_Comm comm);
  
#endif
