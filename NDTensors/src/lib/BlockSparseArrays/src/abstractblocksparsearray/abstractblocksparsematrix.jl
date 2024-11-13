const AbstractBlockSparseMatrix{T} = AbstractBlockSparseArray{T,2}

# SVD is implemented by trying to
# 1. Attempt to find a block-diagonal implementation by permuting
# 2. Fallback to AbstractBlockArray implementation via BlockedArray
function svd(
  A::AbstractBlockSparseMatrix; full::Bool=false, alg::Algorithm=default_svd_alg(A)
)
  T = LinearAlgebra.eigtype(eltype(A))
  A′ = try_to_blockdiagonal(A)

  if isnothing(A′)
    # not block-diagonal, fall back to dense case
    Adense = eigencopy_oftype(A, T)
    return svd!(Adense; full, alg)
  end

  # compute block-by-block and permute back
  A″, (I, J) = A′
  F = svd!(eigencopy_oftype(A″, T); full, alg)
  return SVD(F.U[Block.(I), Block.(J)], F.S, F.Vt)
end