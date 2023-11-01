# Used for dispatch to distinguish from Tensors wrapping TensorStorage.
# Remove once TensorStorage is removed.
const ArrayStorage{T,N} = Union{
  Array{T,N},
  ReshapedArray{T,N},
  SubArray{T,N},
  PermutedDimsArray{T,N},
  StridedView{T,N},
  DiagonalArray{T,N},
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

# TODO: Delete once `Dense` is removed.
function to_arraystorage(x::DenseTensor)
  return tensor(reshape(data(x), size(x)), inds(x))
end

# TODO: Delete once `Diag` is removed.
function to_arraystorage(x::DiagTensor)
  return tensor(DiagonalArray(data(x), size(x)), inds(x))
end

# TODO: Delete once `BlockSparse` is removed.
function to_arraystorage(x::BlockSparseTensor)
  blockinds = map(i -> [blockdim(i, b) for b in 1:nblocks(i)], inds(x))
  blocktype = set_ndims(datatype(x), ndims(x))
  arraystorage = BlockSparseArray{eltype(x),ndims(x),blocktype}(blockinds)
  return tensor(arraystorage, inds(x))
end
