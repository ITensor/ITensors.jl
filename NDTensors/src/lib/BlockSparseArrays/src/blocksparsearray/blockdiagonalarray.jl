# type alias for block-diagonal
using LinearAlgebra: Diagonal

const BlockDiagonal{T,A,Axes,V<:AbstractVector{A}} = BlockSparseMatrix{
  T,A,Diagonal{A,V},Axes
}

function BlockDiagonal(blocks::AbstractVector{<:AbstractMatrix})
  return BlockSparseArray(
    Diagonal(blocks), (blockedrange(size.(blocks, 1)), blockedrange(size.(blocks, 2)))
  )
end

# Cast to block-diagonal implementation if permuted-blockdiagonal
function try_to_blockdiagonal_perm(A)
  inds = map(x -> Int.(Tuple(x)), vec(collect(block_stored_indices(A))))
  I = first.(inds)
  allunique(I) || return nothing
  J = last.(inds)
  p = sortperm(J)
  Jsorted = J[p]
  allunique(Jsorted) || return nothing
  return Block.(I[p], Jsorted)
end

"""
    try_to_blockdiagonal(A)

Attempt to find a permutation of blocks that makes `A` blockdiagonal. If unsuccesful,
returns nothing, otherwise returns both the blockdiagonal `B` as well as the permutation `I, J`.
"""
function try_to_blockdiagonal(A::AbstractBlockSparseMatrix)
  perm = try_to_blockdiagonal_perm(A)
  isnothing(perm) && return perm
  I = first.(Tuple.(perm))
  J = last.(Tuple.(perm))
  diagblocks = map(invperm(I), J) do i, j
    return A[Block(i, j)]
  end
  return BlockDiagonal(diagblocks), perm
end

# SVD implementation
function eigencopy_oftype(A::BlockDiagonal, S)
  diag = map(Base.Fix2(eigencopy_oftype, S), A.blocks.diag)
  return BlockDiagonal(diag)
end

function svd(A::BlockDiagonal; kwargs...)
  return svd!(eigencopy_oftype(A, LinearAlgebra.eigtype(eltype(A))); kwargs...)
end
function svd!(A::BlockDiagonal; full::Bool=false, alg::Algorithm=default_svd_alg(A))
  # TODO: handle full
  F = map(a -> svd!(a; full, alg), blocks(A).diag)
  Us = map(Base.Fix2(getproperty, :U), F)
  Ss = map(Base.Fix2(getproperty, :S), F)
  Vts = map(Base.Fix2(getproperty, :Vt), F)
  return SVD(BlockDiagonal(Us), mortar(Ss), BlockDiagonal(Vts))
end