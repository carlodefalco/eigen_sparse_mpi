
#include <mpi.h>
#include <mpi_alltoallv2.h>

#include <vector>

int
MPI_Alltoallv2 (void ** sendbuf, int *sendcnts, MPI_Datatype sendtype,
                void ** recvbuf, int *recvcnts, MPI_Datatype recvtype,
                MPI_Comm mpicomm)

{
  int mpisize, mpirank;
  MPI_Comm_size (mpicomm, &mpisize);
  MPI_Comm_rank (mpicomm, &mpirank);
  
  std::vector<MPI_Request> reqs;
  for (int ii = 0; ii < mpisize; ++ii)
    {

      if (ii == mpirank) continue; // No communication to self!
      if (recvcnts[ii] > 0)        // we must receive something from rank ii
        {
          int recv_tag = ii   + mpisize * mpirank;
          reqs.resize (reqs.size () + 1);
          MPI_Irecv (recvbuf[ii], recvcnts[ii], recvtype,
                     ii, recv_tag, mpicomm, &(reqs.back ()));
        }

      if (sendcnts[ii] > 0)        // we must send something to rank ii
        {
          int send_tag = mpirank + mpisize * ii;
          reqs.resize (reqs.size () + 1);
          MPI_Isend (sendbuf[ii], sendcnts[ii], sendtype,
                     ii, send_tag, mpicomm, &(reqs.back ()));
        }
    }

  return (MPI_Waitall (reqs.size (), &(reqs[0]), MPI_STATUSES_IGNORE));

}
