# Used for dispatch to distinguish from Tensors wrapping TensorStorage.
# Remove once TensorStorage is removed.
const ArrayLike{T,N} = Union{Array{T,N},ReshapedArray{T,N},SubArray{T,N}}
const MatrixLike{T} = Union{
  Matrix{T},
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
const MatrixOrArrayLike{T} = Union{MatrixLike{T},ArrayLike{T}}

const ArrayLikeTensor{T,N,S,I} = Tensor{T,N,S,I} where {S<:ArrayLike{T,N}}
const MatrixLikeTensor{T,S,I} = Tensor{T,2,S,I} where {S<:MatrixLike{T}}
const MatrixOrArrayLikeTensor{T,S,I} = Tensor{T,N,S,I} where {N,S<:MatrixOrArrayLike{T}}

function getindex(tensor::MatrixOrArrayLikeTensor, I::Integer...)
  return storage(tensor)[I...]
end

function setindex!(tensor::MatrixOrArrayLikeTensor, v, I::Integer...)
  storage(tensor)[I...] = v
  return tensor
end

function contraction_output(
  tensor1::MatrixOrArrayLikeTensor, tensor2::MatrixOrArrayLikeTensor, indsR
)
  tensortypeR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  return NDTensors.similar(tensortypeR, indsR)
end

function contract!(
  tensorR::MatrixOrArrayLikeTensor,
  labelsR,
  tensor1::MatrixOrArrayLikeTensor,
  labels1,
  tensor2::MatrixOrArrayLikeTensor,
  labels2,
)
  contract!(storage(tensorR), labelsR, storage(tensor1), labels1, storage(tensor2), labels2)
  return tensorR
end

function permutedims!(
  output_tensor::MatrixOrArrayLikeTensor, tensor::MatrixOrArrayLikeTensor, perm, f::Function
)
  permutedims!(storage(output_tensor), storage(tensor), perm, f)
  return output_tensor
end

# Linear algebra (matrix algebra)
Base.adjoint(tens::MatrixLikeTensor) = tensor(adjoint(storage(tens)), reverse(inds(tens)))

function LinearAlgebra.mul!(C::MatrixLikeTensor, A::MatrixLikeTensor, B::MatrixLikeTensor)
  mul!(storage(C), storage(A), storage(B))
  return C
end

function LinearAlgebra.svd(tens::MatrixLikeTensor)
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

array(tensor::MatrixOrArrayLikeTensor) = storage(tensor)

# Combiner
function contraction_output(
  tensor1::MatrixOrArrayLikeTensor, tensor2::CombinerTensor, indsR
)
  tensortypeR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  return NDTensors.similar(tensortypeR, indsR)
end
