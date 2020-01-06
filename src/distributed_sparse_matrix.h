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
  
  using seq_type = Eigen::SparseMatrix<double,Eigen::RowMajor>;

  int
  memory_estimate ()
  {
    int mem = 0;
    
    mem += sizeof (mapped);
    mem += sizeof (decltype(this->ranges)::value_type) * this->ranges.size ();
    mem += sizeof (decltype(this->rank_nnz)::value_type) * this->rank_nnz.size ();

    for (auto& i : this->val_buffers)
      mem += sizeof (decltype(this->val_buffers)::key_type) + sizeof (decltype(this->val_buffers)::mapped_type::value_type) * i.second.size ();

    for (auto& i : this->col_buffers)
      mem += sizeof (decltype(this->col_buffers)::key_type) + sizeof (decltype(this->col_buffers)::mapped_type::value_type) * i.second.size ();

    for (auto& i : this->row_buffers)
      mem += sizeof (decltype(this->row_buffers)::key_type) + sizeof (decltype(this->row_buffers)::mapped_type::value_type) * i.second.size ();

    mem += sizeof (decltype(this->non_local.prc_ptr)::value_type) * this->non_local.prc_ptr.size ();
    mem += sizeof (decltype(this->non_local.row_ind)::value_type) * this->non_local.row_ind.size ();
    mem += sizeof (decltype(this->non_local.col_ind)::value_type) * this->non_local.col_ind.size ();
    mem += sizeof (decltype(this->non_local.a)::value_type) * this->non_local.a.size ();

    mem += sizeof (mpirank);
    mem += sizeof (mpisize);
    mem += sizeof (comm);
    mem += sizeof (is);
    mem += sizeof (ie);

    mem += this->nonZeros () * sizeof (Scalar) * sizeof (StorageIndex) + this->outerSize () * sizeof (StorageIndex);
    
    return mem;
  }

  // the number of rows is inferred from the
  // ranges, the nuber of columns must be specified
  // it would be probably better to enforce this
  // somehow? add a required c'tor parameter?
  // error out on assemble if cols not set?
  void
  set_cols (int cols_)
  {
    this->resize (this->rows (), cols_);
  }
  
  void
  set_ranges (int is_, int ie_);

  void
  set_ranges (int num_owned_);

  distributed_sparse_matrix (int is_, int ie_,
                             MPI_Comm comm_ = MPI_COMM_WORLD)
    : distributed_sparse_matrix (comm_)
  { set_ranges (is_, ie_); }

  distributed_sparse_matrix (MPI_Comm comm_ = MPI_COMM_WORLD)
    : seq_type::SparseMatrix (), comm (comm_), mapped (false)
  { }

  void
  remap ();

  void
  assemble ();

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
