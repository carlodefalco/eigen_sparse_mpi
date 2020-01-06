#include <array>
#include <cmath>
#include <iostream>
#include <distributed_sparse_matrix.h>

int
main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);
  int rank, size;
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  
  distributed_sparse_matrix A;
  int n =  std::sqrt (size);

  // we want to create a partition into n * n
  // subregions and assign each region to one
  // rank, for this to work n*n must equal size
  // if this is not the case we just fail and exit
  // a more graceful exit would actually be nicer
  assert (n*n == size);

  // the number of grid cells per rank is nN * nN
  // the total number of grind cells is N * N, the
  // total number of grid nodes is ndofs
  int nN =  60;
  int N = nN * n;
  int ndofs = ((N+1)*(N+1));
  std::cout << "ndofs = " << ndofs << std::endl;

  // the number of rows is inferred from the
  // ranges, the nuber of columns must be specified
  A.set_cols (ndofs);
  if (rank < size - 1)
    A.set_ranges (ndofs / size);
  else
    A.set_ranges (ndofs - (size - 1) * (ndofs / size));
  
  int partrow = rank / n;
  int partcol = rank % n;

  
  
  for (int ii = 0; ii < nN; ++ii)
    for (int jj = 0; jj < nN; ++jj)
      {        
        int kk = jj + nN * partcol + (N + 1) * (ii + nN * partrow);               
        std::array<int, 4> conn = {kk, kk+1, kk+N+1, kk+N+2};
        std::array<std::array<double, 4>, 4> loc =
          {  .25, -.125, -.125,    .0,
             -.125,   .25,    .0, -.125,
             -.125,    .0,   .25, -.125,
             .0, -.125,  -.125,  .25};

        for (int irow = 0; irow < 4; ++irow)
          for (int jcol = 0; jcol < 4; ++jcol)
            {
              A.coeffRef(conn[irow],conn[jcol]) += loc[irow][jcol];
            }
      }
 
  
  
  
  A.assemble ();

  for (int ii = 0; ii < size; ii++)
    {
      if (ii == rank)
        {
          //std::cout << A << std::endl;
          std::cout << "rank " << rank << " done" << std::endl;
        }
      MPI_Barrier (MPI_COMM_WORLD);
    }
  
  MPI_Finalize ();
  return 0;
}
