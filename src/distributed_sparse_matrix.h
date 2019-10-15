#ifndef HAVE_DISTRIBUTED_SPARSE_MATRIX_H
#  define HAVE_DISTRIBUTED_SPARSE_MATRIX_H

#include <Eigen/Sparse>
#include <mpi.h>

class
distributed_sparse_matrix
  : public Eigen::SparseMatrix<double,Eigen::RowMajor>
{
private :

  void
  non_local_csr ();

  int is, ie;
  MPI_Comm comm;
  int mpirank, mpisize;

  struct
  non_local_t
  {
    std::vector<int> prc_ptr, row_ind, col_ind;
    std::vector<double> a;
  } non_local;

  std::map<int, std::vector<int>> row_buffers;
  std::map<int, std::vector<int>> col_buffers;
  std::map<int, std::vector<double>> val_buffers;

  std::vector<int> ranges;
  std::vector<int> rank_nnz;

  bool mapped;

public :

  void
  print_non_local ()
  {
    std::cout << "prc_ptr : \n";
    for ( auto& i : non_local.prc_ptr)
      std::cout << i << "\t";
    std::cout << std::endl << std::endl;
    for ( auto& i : non_local.row_ind)
      std::cout << i << "\t";
    std::cout << std::endl << std::endl;
    for ( auto& i : non_local.col_ind)
      std::cout << i << "\t";
    std::cout << std::endl << std::endl;
    for ( auto& i : non_local.a)
      std::cout << i << "\t";
    std::cout << std::endl << std::endl;
  };
  
  using seq_type = Eigen::SparseMatrix<double,Eigen::RowMajor>;

  void
  set_ranges (int is_, int ie_);

  void
  set_ranges (int num_owned_);

  distributed_sparse_matrix (int is_, int ie_,
                             MPI_Comm comm_ = MPI_COMM_WORLD)
    : distributed_sparse_matrix ()
  { set_ranges (is_, ie_); }

  distributed_sparse_matrix (MPI_Comm comm_ = MPI_COMM_WORLD)
    : seq_type::SparseMatrix (), comm (comm_), mapped (false)
  { }

  void
  remap ();

  void
  assemble ();

  void
  csr (std::vector<double> &a,
       std::vector<int> &col,
       std::vector<int> &row,
       int base = 0,
       bool flag = false);

  void
  csr_update (std::vector<double> &a,
              const std::vector<int> &col_ind,
              const std::vector<int> &row_ptr,
              int base = 0,
              bool flag = false);

  void
  aij (std::vector<double> &a,
       std::vector<int> &i,
       std::vector<int> &j,
       int base = 0,
       bool flag = false);

  int
  owned_nnz ();

  int
  range_start ()
  { return is; };

  int
  range_end ()
  { return ie; };

};

#endif // HAVE_DISTRIBUTED_SPARSE_MATRIX_H
