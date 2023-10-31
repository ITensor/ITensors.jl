# The output must be initialized as zero since it is sparse, cannot be undefined
# Overspecifying types to fix ambiguity error.
# TODO: Rewrite in terms of `DiagonalArray` and `Array`, not `Tensor`.
function contraction_output(
  T1::Tensor{T,N,<:DiagonalArray{T,N,<:AbstractVector{T}}}, T2::ArrayStorageTensor, indsR
) where {T,N}
  return zero_contraction_output(T1, T2, indsR)
end

# TODO: Rewrite in terms of `DiagonalArray` and `Array`, not `Tensor`.
function contraction_output(
  T1::ArrayStorageTensor, T2::Tensor{T,N,<:DiagonalArray{T,N,<:AbstractVector{T}}}, indsR
) where {T,N}
  return contraction_output(T2, T1, indsR)
end

# Overspecifying types to fix ambiguity error.
# TODO: Rewrite in terms of `DiagonalArray`, not `Tensor`.
function contraction_output(
  tensor1::Tensor{T1,N1,<:DiagonalArray{T1,N1,<:AbstractVector{T1}}},
  tensor2::Tensor{T2,N2,<:DiagonalArray{T2,N2,<:AbstractVector{T2}}},
  indsR,
) where {T1,N1,T2,N2}
  return zero_contraction_output(tensor1, tensor2, indsR)
end
