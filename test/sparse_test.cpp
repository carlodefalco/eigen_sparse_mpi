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
  A.resize (6, 7);
  A.reserve (21);
  A.set_ranges (3);
  
  std::vector<int> ir = {0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5};
  std::vector<int> ic = {0, 1, 2, 5, 6, 0, 1, 4, 6, 0, 3, 4, 1, 4, 6, 3, 5, 0, 2, 4, 5};
  std::vector<double> a  = {1., 2., 3., 1., 1., 2., 3., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3., 3};
  
  for (int ii = 0; ii < a.size (); ++ii)
    {
      A.insert (ir[ii], ic[ii]) = a[ii];
    }

  if (0 == rank)
    std::cout << "**** BEFORE ASSEMBLY ***" << std::endl;
  for (int ii = 0; ii < size; ++ii)
    {
      if (ii == rank)
        {
          std::cout << "rank " << rank << std::endl;
          std::cout << A;
        }
      MPI_Barrier (MPI_COMM_WORLD);
    }
  
  A.assemble ();  

  if (0 == rank)
    std::cout << "**** AFTER ASSEMBLY ***" << std::endl;
  for (int ii = 0; ii < size; ++ii)
    {
      if (ii == rank)
        {
          std::cout << "rank " << rank << std::endl;
          std::cout << A;
        }
      MPI_Barrier (MPI_COMM_WORLD);
    }

  MPI_Finalize ();
  return 0;
}
