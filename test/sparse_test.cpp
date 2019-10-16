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
  int nN =  20;
  int N = nN * n;
  int ndofs = ((N+1)*(N+1));
  /*
  std::cout << "rank " << rank
            << ", n = " << n
            << ", nN = " << nN
            << ", N = " << N 
            << ", ndofs =" << ndofs << "\n";
  */
  A.resize (ndofs, ndofs);
  if (rank < size - 1)
    A.set_ranges (ndofs / size);
  else
    A.set_ranges (ndofs - (size - 1) * (ndofs / size));
  
  int partrow = rank / n;
  int partcol = rank % n;

  /*
    for (int nn = 0; nn < size ; ++nn)
    {
    if (rank == nn)
    { 
    std::cout << "rank " << rank
    << ", partrow = "
    << partrow << ", partcol = "
    << partcol << ", A.rows() = " << A.rows()
    << ", A.cols() = " << A.cols()
    << std::endl;
  */
  for (int ii = 0; ii < nN; ++ii)    
    for (int jj = 0; jj < nN; ++jj)
      {        
        int kk = jj + nN * partcol + (N + 1) * (ii + nN * partrow);               
        //std::cout << "kk = " << kk << std::endl;
        std::array<int, 4> conn = {kk, kk+1, kk+N+1, kk+N+2};
        std::array<std::array<double, 4>, 4> loc =
          {  .25, -.125, -.125,    .0,
             -.125,   .25,    .0, -.125,
             -.125,    .0,   .25, -.125,
             .0, -.125,  -.125,  .25};

        for (int irow = 0; irow < 4; ++irow)
          for (int jcol = 0; jcol < 4; ++jcol)
            {
              //std::cout << "(" << conn[irow] << "/" << A.rows() << ", " << conn[jcol] << "/" << A.cols() << ")" << std::endl;
              A.coeffRef(conn[irow],conn[jcol]) += loc[irow][jcol];
            }
      }
  /*
    }
    MPI_Barrier (MPI_COMM_WORLD);
    }
  */
  
  A.assemble ();

  //  std::cout << A << std::endl;
  for (int nn = 0; nn < size ; ++nn)
    {
      if (rank == nn)
        { 

          std::cout << "rank " << rank << ", memory = "
                    << A.memory_estimate ()  << ", nnz = "
                    << A.nonZeros () << ", nrows = "
                    << A.outerSize () << std::endl;
        }
      MPI_Barrier (MPI_COMM_WORLD);
    }

  std::cout << "rank " << rank << " done" << std::endl;
  MPI_Finalize ();
  return 0;
}
