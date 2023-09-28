# Used for dispatch to distinguish from Tensors wrapping TensorStorage.
# Remove once TensorStorage is removed.
const ArrayStorage{T,N} = Union{
  Array{T,N},ReshapedArray{T,N},SubArray{T,N},PermutedDimsArray{T,N},StridedView{T,N}
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

const ArrayStorageTensor{T,N,S,I} = Tensor{T,N,S,I} where {S<:ArrayStorage{T,N}}
const MatrixStorageTensor{T,S,I} = Tensor{T,2,S,I} where {S<:MatrixStorage{T}}
const MatrixOrArrayStorageTensor{T,S,I} =
  Tensor{T,N,S,I} where {N,S<:MatrixOrArrayStorage{T}}

function getindex(tensor::MatrixOrArrayStorageTensor, I::Integer...)
  return storage(tensor)[I...]
end

function setindex!(tensor::MatrixOrArrayStorageTensor, v, I::Integer...)
  storage(tensor)[I...] = v
  return tensor
end

function contraction_output(
  tensor1::MatrixOrArrayStorageTensor, tensor2::MatrixOrArrayStorageTensor, indsR
)
  tensortypeR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  return NDTensors.similar(tensortypeR, indsR)
end

function contract!(
  tensorR::MatrixOrArrayStorageTensor,
  labelsR,
  tensor1::MatrixOrArrayStorageTensor,
  labels1,
  tensor2::MatrixOrArrayStorageTensor,
  labels2,
)
  contract!(storage(tensorR), labelsR, storage(tensor1), labels1, storage(tensor2), labels2)
  return tensorR
end

function permutedims!(
  output_tensor::MatrixOrArrayStorageTensor,
  tensor::MatrixOrArrayStorageTensor,
  perm,
  f::Function,
)
  permutedims!(storage(output_tensor), storage(tensor), perm, f)
  return output_tensor
end

# Linear algebra (matrix algebra)
function Base.adjoint(tens::MatrixStorageTensor)
  return tensor(adjoint(storage(tens)), reverse(inds(tens)))
end

function LinearAlgebra.mul!(
  C::MatrixStorageTensor, A::MatrixStorageTensor, B::MatrixStorageTensor
)
  mul!(storage(C), storage(A), storage(B))
  return C
end

function LinearAlgebra.svd(tens::MatrixStorageTensor)
  F = svd(storage(tens))
  U, S, V = F.U, F.S, F.Vt
  i, j = inds(tens)
  # TODO: Make this more general with a `similar_ind` function,
  # so the dimension can be determined from the length of `S`.
  min_ij = dim(i) ≤ dim(j) ? i : j
  α = sim(min_ij) # similar_ind(i, space(S))
  β = sim(min_ij) # similar_ind(i, space(S))
  Utensor = tensor(U, (i, α))
  # TODO: Remove conversion to `Matrix` to make more general.
  # Used for now to avoid introducing wrapper types.
  Stensor = tensor(Diagonal(S), (α, β))
  Vtensor = tensor(V, (β, j))
  return Utensor, Stensor, Vtensor, Spectrum(nothing, 0.0)
end

array(tensor::MatrixOrArrayStorageTensor) = storage(tensor)

# Combiner
function contraction_output(
  tensor1::MatrixOrArrayStorageTensor, tensor2::CombinerTensor, indsR
)
  tensortypeR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  return NDTensors.similar(tensortypeR, indsR)
end
