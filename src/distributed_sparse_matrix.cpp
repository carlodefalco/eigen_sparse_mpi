#include <distributed_sparse_matrix.h>
#include <mpi_alltoallv2.h>

void
distributed_sparse_matrix::set_ranges (int is_, int ie_)
{
  is = is_; ie = ie_;
  MPI_Comm_rank (comm, &mpirank);
  MPI_Comm_size (comm, &mpisize);

  /// Gather ranges
  ranges.assign (mpisize + 1, 0);
  MPI_Allgather (&ie, 1, MPI_INT, &(ranges[1]), 1, MPI_INT, comm);
  this->resize (ranges.back (), this->cols ());
}

void
distributed_sparse_matrix::set_ranges (int num_owned_)
{
  MPI_Comm_size (comm, &mpisize);
  MPI_Comm_rank (comm, &mpirank);

  /// Gather ranges
  ranges.assign (mpisize + 1, 0);
  MPI_Allgather (&num_owned_, 1, MPI_INT, &(ranges[1]),
                 1, MPI_INT, comm);
  for (auto irank = 0; irank < mpisize; ++irank)
    ranges[irank+1] += ranges[irank];
  
  is = ranges[mpirank];
  ie = ranges[mpirank+1];
  this->resize (ranges.back (), this->cols ());
}


/*

The example below shows the contents of the non_local struct
for a 6x7 matrix partitioned in 2-row blocks over 3 ranks

MATRIX :

[1 2 3 0 0 1 1
 2 3 0 0 1 0 1
 2 0 0 1 1 0 0
 0 1 0 0 1 0 1
 0 0 0 1 0 1 0
 1 0 1 0 3 3 0]

CSR :

r : 0         5       9     12     15   17       21
c : 0 1 2 5 6 0 1 4 6 0 3 4  1 4 6  3 5  0 2 4 5
a : 1 2 3 1 1 2 3 1 1 2 1 1  1 1 1  1 1  1 1 3 3


NLCSR, size = 3, rank = 1 :

p : 0                 9 9           15
r : 0 0 0 0 0 1 1 1 1   4 4 5 5 5 5
c : 0 1 2 5 6 0 1 4 6   3 5 0 2 4 5
a : 1 2 3 1 1 2 3 1 1   1 1 1 1 3 3

*/

void
distributed_sparse_matrix::non_local_csr ()
{
  non_local.row_ind.clear ();
  non_local.col_ind.clear ();
  non_local.a.clear ();

  if (! this->isCompressed ())
    this->makeCompressed ();

  const auto ir = this->outerIndexPtr ();
  const auto ic = this->innerIndexPtr ();
  const auto xa = this->valuePtr ();
  
  non_local.prc_ptr.assign (this->mpisize + 1, 0);
  std::vector<int> proc_nnz (this->mpisize);

  for (int i_proc = 0; i_proc < this->mpisize; ++i_proc)
    {
     
      if (i_proc != this->mpirank)
        proc_nnz[i_proc] = ir[ranges[i_proc+1]] -
          ir[ranges[i_proc]];
      else
        proc_nnz[i_proc] = 0;

      
      non_local.prc_ptr[i_proc + 1] =
        non_local.prc_ptr[i_proc] + proc_nnz[i_proc];
      
    }

  
  int non_local_nnz = non_local.prc_ptr[this->mpisize];

  non_local.row_ind.assign (non_local_nnz, 0);
  non_local.col_ind.assign (non_local_nnz, 0);
  non_local.a.assign (non_local_nnz, 0.0);

  int jr = 0, jc = 0, c = 0, cc = 0;
  for (int i_proc = 0; i_proc < this->mpisize; ++i_proc)
    {      
      if (i_proc != this->mpirank)
        {
          for (jr = this->ranges[i_proc];
               jr < this->ranges[i_proc+1]; ++jr)
            {
              for (jc = ir[jr]; jc < ir[jr+1]; ++jc)
                {
                  non_local.row_ind[c++] = jr;
                }
            }
          std::copy (ic + ir[this->ranges[i_proc]],
                     ic + ir[this->ranges[i_proc]] +
                     proc_nnz[i_proc],
                     non_local.col_ind.begin () + cc);
         
          cc += proc_nnz[i_proc];
        }
    }
}

int
distributed_sparse_matrix::owned_nnz ()
{
  if (! this->isCompressed ())
    this->makeCompressed ();

  const auto ir = this->outerIndexPtr ();
  return (ir[this->mpirank+1]-ir[this->mpirank]);
}



void
distributed_sparse_matrix::remap ()
{
  non_local_csr ();

  /// Distribute buffer sizes
  rank_nnz.assign (mpisize, 0);
  for (int ii = 0; ii < mpisize; ++ii)
    rank_nnz[ii] = non_local.prc_ptr[ii+1] - non_local.prc_ptr[ii];
  MPI_Alltoall (MPI_IN_PLACE, 1, MPI_INT, &(rank_nnz[0]),
                1, MPI_INT, comm);

  /// Allocate buffers
  for (int ii = 0; ii < mpisize; ++ii)
    if ((rank_nnz[ii] > 0) && (ii != mpirank))
      {
        row_buffers[ii].resize (rank_nnz[ii]);
        col_buffers[ii].resize (rank_nnz[ii]);
        val_buffers[ii].resize (rank_nnz[ii]);
      }


  /// Communicate overlap regions
  void * sendbuf[mpisize];
  int sendcnts[mpisize];
  void * recvbuf[mpisize];
  int recvcnts[mpisize];

  /// 1) communicate row_ptr
  for (int ii = 0; ii < mpisize; ++ii)
    {
      recvbuf[ii]  = &(row_buffers[ii][0]);
      recvcnts[ii] =   row_buffers[ii].size ();
      sendbuf[ii]  = &(non_local.row_ind[non_local.prc_ptr[ii]]);
      sendcnts[ii] =   non_local.prc_ptr[ii+1] -
        non_local.prc_ptr[ii];
    }

  MPI_Alltoallv2 (sendbuf, sendcnts, MPI_INT,
                  recvbuf, recvcnts, MPI_INT, comm);

  /// 2) communicate col_ind
  for (int ii = 0; ii < mpisize; ++ii)
    {
      recvbuf[ii]  = &(col_buffers[ii][0]);
      recvcnts[ii] =   col_buffers[ii].size ();
      sendbuf[ii]  = &(non_local.col_ind[non_local.prc_ptr[ii]]);
      sendcnts[ii] =   non_local.prc_ptr[ii+1] -
        non_local.prc_ptr[ii];
    }

  MPI_Alltoallv2 (sendbuf, sendcnts, MPI_INT,
                  recvbuf, recvcnts, MPI_INT, comm);
  mapped = true;
}


void
distributed_sparse_matrix::assemble ()
{

  if (! mapped)
    remap ();

  non_local.a.resize (non_local.row_ind.size ());
  for (int j = 0; j < non_local.col_ind.size (); ++j)
    non_local.a[j] = this->coeffRef (non_local.row_ind[j],
                                     non_local.col_ind[j]);

  /// 3) communicate values
  void * sendbuf[mpisize];
  int sendcnts[mpisize];
  void * recvbuf[mpisize];
  int recvcnts[mpisize];

  for (int ii = 0; ii < mpisize; ++ii)
    {
      recvbuf[ii]  = &(val_buffers[ii][0]);
      recvcnts[ii] =   val_buffers[ii].size ();
      sendbuf[ii]  = &(non_local.a[non_local.prc_ptr[ii]]);
      sendcnts[ii] =   non_local.prc_ptr[ii+1] -
        non_local.prc_ptr[ii];
    }

  MPI_Alltoallv2 (sendbuf, sendcnts, MPI_DOUBLE,
                  recvbuf, recvcnts, MPI_DOUBLE, comm);

  /// 4) insert communicated values into sparse_matrix
  for (int ii = 0; ii < mpisize; ++ii) // loop over ranks
    if (ii != mpirank)
      for (int kk = 0; kk < rank_nnz[ii]; ++kk)
        this->coeffRef(row_buffers[ii][kk],col_buffers[ii][kk])
          += val_buffers[ii][kk];


  /// 5) zero out communicated values
  for (int iprc = 0; iprc < mpisize; ++iprc)
    if (iprc != mpirank)
      {
        for (int ii = non_local.prc_ptr[iprc];
             ii < non_local.prc_ptr[iprc+1]; ++ii)
          // the following will throw if we try to access
          // an element which does not exist yet!
          this->coeffRef (non_local.row_ind[ii],
                          non_local.col_ind[ii]) = 0.0;
      }

}


