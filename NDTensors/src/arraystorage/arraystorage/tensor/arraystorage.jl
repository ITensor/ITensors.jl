const ArrayStorageTensor{T,N,S,I} = Tensor{T,N,S,I} where {S<:ArrayStorage{T,N}}
const MatrixStorageTensor{T,S,I} = Tensor{T,2,S,I} where {S<:MatrixStorage{T}}
const MatrixOrArrayStorageTensor{T,S,I} =
  Tensor{T,N,S,I} where {N,S<:MatrixOrArrayStorage{T}}

function Tensor(storage::MatrixOrArrayStorageTensor, inds::Tuple)
  return Tensor(NeverAlias(), storage, inds)
end

function Tensor(as::AliasStyle, storage::MatrixOrArrayStorage, inds::Tuple)
  return Tensor{eltype(storage),length(inds),typeof(storage),typeof(inds)}(
    as, storage, inds
  )
end

array(tensor::MatrixOrArrayStorageTensor) = storage(tensor)

# Linear algebra (matrix algebra)
function Base.adjoint(tens::MatrixStorageTensor)
  return tensor(adjoint(storage(tens)), reverse(inds(tens)))
end

# Linear algebra (matrix algebra)
function LinearAlgebra.Hermitian(tens::MatrixStorageTensor)
  return tensor(Hermitian(storage(tens)), inds(tens))
end
