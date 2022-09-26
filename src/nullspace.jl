#
# NDTensors functionality
#

# XXX: generalize this function
function _getindex(T::DenseTensor{ElT,N}, I1::Colon, I2::UnitRange{Int64}) where {ElT,N}
  A = array(T)[I1, I2]
  return tensor(Dense(vec(A)), setdims(inds(T), size(A)))
end

function getblock_preserve_qns(T::Tensor, b::Block)
  # TODO: make `T[b]` preserve QNs
  Tb = T[b]
  indsTb = getblock.(inds(T), Tuple(b)) .* dir.(inds(T))
  return ITensors.setinds(Tb, indsTb)
end

function blocksparsetensor(blocks::Dict{B,TB}) where {B,TB}
  b1, Tb1 = first(pairs(blocks))
  N = length(b1)
  indstypes = typeof.(inds(Tb1))
  blocktype = eltype(Tb1)
  indsT = getindex.(indstypes)
  # Determine the indices from the blocks
  for (b, Tb) in pairs(blocks)
    indsTb = inds(Tb)
    for n in 1:N
      bn = b[n]
      indsTn = indsT[n]
      if bn > length(indsTn)
        resize!(indsTn, bn)
      end
      indsTn[bn] = indsTb[n]
    end
  end
  T = BlockSparseTensor(blocktype, indsT)
  for (b, Tb) in pairs(blocks)
    if !isempty(Tb)
      T[b] = Tb
    end
  end
  return T
end

default_atol(A::AbstractArray) = 0.0
function default_rtol(A::AbstractArray, atol::Real)
  return (min(size(A, 1), size(A, 2)) * eps(real(float(one(eltype(A)))))) * iszero(atol)
end

function _nullspace_hermitian(
  M::DenseTensor; atol::Real=default_atol(M), rtol::Real=default_rtol(M, atol)
)
  # TODO: try this version
  #D, U = eigen(Hermitian(M))
  Dᵢₜ, Uᵢₜ = eigen(itensor(M); ishermitian=true)
  D = tensor(Dᵢₜ)
  U = tensor(Uᵢₜ)
  tol = max(atol, abs(D[1, 1]) * rtol)
  indstart = sum(d -> abs(d) .> tol, storage(D)) + 1
  indstop = lastindex(U, 2)
  Nb = _getindex(U, :, indstart:indstop)
  return Nb
end

function _nullspace_hermitian(
  M::BlockSparseTensor; atol::Real=default_atol(M), rtol::Real=default_rtol(M, atol)
)
  tol = atol
  # TODO: try this version
  # Insert any missing diagonal blocks
  insert_diag_blocks!(M)
  #D, U = eigen(Hermitian(M))
  Dᵢₜ, Uᵢₜ = eigen(itensor(M); ishermitian=true)
  D = tensor(Dᵢₜ)
  U = tensor(Uᵢₜ)
  nullspace_blocks = Dict()
  for bU in nzblocks(U)
    bM = Block(bU[1], bU[1])
    bD = Block(bU[2], bU[2])
    # Assume sorted from largest to smallest
    tol = max(atol, abs(D[bD][1, 1]) * rtol)
    indstart = sum(d -> abs(d) .> tol, storage(D[bD])) + 1
    Ub = getblock_preserve_qns(U, bU)
    indstop = lastindex(Ub, 2)
    # Drop zero dimensional blocks
    Nb = _getindex(Ub, :, indstart:indstop)
    nullspace_blocks[bU] = Nb
  end
  return blocksparsetensor(nullspace_blocks)
end

function LinearAlgebra.nullspace(M::Hermitian{<:Number,<:Tensor}; kwargs...)
  return _nullspace_hermitian(parent(M); kwargs...)
end

#
# QN functionality
#

function setdims(t::NTuple{N,Pair{QN,Int}}, dims::NTuple{N,Int}) where {N}
  return first.(t) .=> dims
end

function setdims(t::NTuple{N,Index{Int}}, dims::NTuple{N,Int}) where {N}
  return dims
end

function getblock(i::Index, n::Integer)
  return ITensors.space(i)[n]
end

# Make `Pair{QN,Int}` act like a regular `dim`
NDTensors.dim(qnv::Pair{QN,Int}) = last(qnv)

Base.:*(qnv::Pair{QN,Int}, d::ITensors.Arrow) = qn(qnv) * d => dim(qnv)

#
# ITensors functionality
#

# Reshape into an order-2 ITensor
matricize(T::ITensor, inds::Index...) = matricize(T, inds)

function matricize(T::ITensor, inds)
  left_inds = commoninds(T, inds)
  right_inds = uniqueinds(T, inds)
  return matricize(T, left_inds, right_inds)
end

function matricize(T::ITensor, left_inds, right_inds)
  CL = combiner(left_inds; dir=ITensors.Out, tags="CL")
  CR = combiner(right_inds; dir=ITensors.In, tags="CR")
  M = (T * CL) * CR
  return M, CL, CR
end

function nullspace(::Order{2}, M::ITensor, left_inds, right_inds; tags="n", kwargs...)
  @assert order(M) == 2
  M² = prime(dag(M), right_inds) * M
  M² = permute(M², right_inds'..., right_inds...)
  M²ₜ = tensor(M²)
  Nₜ = nullspace(Hermitian(M²ₜ); kwargs...)
  indsN = (Index(ind(Nₜ, 1); dir=ITensors.Out), Index(ind(Nₜ, 2); dir=ITensors.Out, tags))
  N = itensor(ITensors.setinds(Nₜ, indsN))
  # Make the index match the input index
  Ñ = replaceinds(N, (ind(N, 1),) => right_inds)
  return Ñ
end

"""
    nullspace(T::ITensor, left_inds...; tags="n", atol=1E-12, kwargs...)

Viewing the ITensor `T` as a matrix with the provided `left_inds` viewed
as the row space and remaining indices viewed as the right indices or column space,
the `nullspace` function computes the right null space. That is, it will return
a tensor `N` acting on the right indices of `T` such that `T*N` is zero.
The returned tensor `N` will also have a new index with the label "n" which
indexes through the 'vectors' in the null space.

For example, if `T` has the indices `i,j,k`, calling
`N = nullspace(T,i,k)` returns `N` with index `j` such that

           ___       ___
      i --|   |     |   |
          | T |--j--| N |--n  ≈ 0
      k --|   |     |   |
           ---       ---

The index `n` can be obtained by calling
`n = uniqueindex(N,T)`

Note that the implementation of this function is subject to change in the future, in
which case the precise `atol` value that gives a certain null space size may change
in future versions of ITensor.

Keyword arguments:

  - `atol::Float64=1E-12` - singular values of T†*T below this value define the null space
  - `tags::String="n"` - choose the tags of the index selecting elements of the null space
"""
function nullspace(T::ITensor, is...; tags="n", atol=1E-12, kwargs...)
  M, CL, CR = matricize(T, is...)
  @assert order(M) == 2
  cL = commoninds(M, CL)
  cR = commoninds(M, CR)
  N₂ = nullspace(Order(2), M, cL, cR; tags, atol, kwargs...)
  return N₂ * CR
end
