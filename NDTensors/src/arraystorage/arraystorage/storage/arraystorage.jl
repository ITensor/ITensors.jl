# Used for dispatch to distinguish from Tensors wrapping TensorStorage.
# Remove once TensorStorage is removed.
const ArrayStorage{T,N} = Union{
  Array{T,N},
  ReshapedArray{T,N},
  SubArray{T,N},
  PermutedDimsArray{T,N},
  StridedView{T,N},
  BlockSparseArray{T,N},
}

const MatrixStorage{T} = Union{
  ArrayStorage{T,2},
  Transpose{T},
  Adjoint{T},
  Symmetric{T},
  Hermitian{T},
  UpperTriangular{T},
  LowerTriangular{T},
  UnitUpperTriangular{T},
  UnitLowerTriangular{T},
  Diagonal{T},
}

const MatrixOrArrayStorage{T} = Union{MatrixStorage{T},ArrayStorage{T}}
