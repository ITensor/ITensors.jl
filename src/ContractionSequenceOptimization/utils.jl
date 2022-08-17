
#
# General helper functionality
#

#
# Operations for tree data structures
#

"""
    deepmap(f, tree; filter=(x -> x isa AbstractArray))

Recursive map on a tree-like data structure.
`filter` is a function that returns `true` if the iteration
should continue `false` if the iteration should
stop (for example, because we are at a leaf and the function
`f` should be applied).

```julia
julia> deepmap(x -> 2x, [1, [2, [3, 4]]])
[2, [4, [6, 8]]]

julia> deepmap(x -> 2 .* x, [1, (2, [3, 4])])
2-element Vector{Any}:
 2
  (4, [6, 8])

julia> deepmap(x -> 3x, [[1 2; 3 4], [5 6; 7 8]])
2-element Vector{Matrix{Int64}}:
 [3 6; 9 12]
 [15 18; 21 24]

julia> deepmap(x -> 2x, (1, (2, (3, 4))); filter=(x -> x isa Tuple))
(2, (4, (6, 8)))
```
"""
function deepmap(f, tree; filter=(x -> x isa AbstractArray))
  return filter(tree) ? map(t -> deepmap(f, t; filter=filter), tree) : f(tree)
end

# 
# Contracting index sets and getting costs
#

# TODO: make a type:
# ShortBitSet{T <: Unsigned}
#   data::T
# end
#
# That is like a BitSet/BitVector but with a maximum set size.
# Aliases such as:
#
# const SBitSet64 = SBitSet{UInt64}
#
# would be helpful to specify a BitSet with a maximum number of
# 64 elements in the set.
# (See https://discourse.julialang.org/t/parse-an-array-of-bits-bitarray-to-an-integer/42361/11).

# Previously we used the definition in NDTensors:
#import NDTensors: dim
import ITensors: dim

# `is` could be Vector{Int} for BitSet
function dim(is::IndexSetT, ind_dims::Vector) where {IndexSetT<:Union{Vector{Int},BitSet}}
  dim = one(eltype(ind_dims))
  for i in is
    dim *= ind_dims[i]
  end
  return dim
end

function dim(is::Unsigned, ind_dims::Vector{DimT}) where {DimT}
  _isemptyset(is) && return one(eltype(ind_dims))
  dim = one(eltype(ind_dims))
  i = 1
  @inbounds while !iszero(is)
    if isodd(is)
      # TODO: use Base.Checked.mul_with_overflow
      dim, overflow = Base.Checked.mul_with_overflow(dim, ind_dims[i])
      overflow && return zero(DimT)
    end
    is = is >> 1
    i += 1
  end
  return dim
end

function contraction_cost(indsTᵃ::BitSet, indsTᵇ::BitSet, dims::Vector)
  indsTᵃTᵇ = _symdiff(indsTᵃ, indsTᵇ)
  dim_a = dim(indsTᵃ, dims)
  dim_b = dim(indsTᵇ, dims)
  dim_ab = dim(indsTᵃTᵇ, dims)
  # Perform the sqrt first to avoid overflow.
  # Alternatively, use a larger integer type.
  cost = round(Int, sqrt(dim_a) * sqrt(dim_b) * sqrt(dim_ab))
  return cost, indsTᵃTᵇ
end

function contraction_cost(
  indsTᵃ::IndexSetT, indsTᵇ::IndexSetT, dims::Vector
) where {IndexSetT<:Unsigned}
  unionTᵃTᵇ = _union(indsTᵃ, indsTᵇ)
  cost = dim(unionTᵃTᵇ, dims)
  indsTᵃTᵇ = _setdiff(unionTᵃTᵇ, _intersect(indsTᵃ, indsTᵇ))
  return cost, indsTᵃTᵇ
end

#
# Convert indices into unique integer labels
#

function contraction_labels!(labels1, labels2, is1, is2)
  nextlabel = 1
  nextlabel = common_contraction_labels!(labels1, labels2, is1, is2, nextlabel)
  nextlabel = uncommon_contraction_labels!(labels1, is1, nextlabel)
  nextlabel = uncommon_contraction_labels!(labels2, is2, nextlabel)
  return labels1, labels2
end

# Compute the common contraction labels and return the next label
function common_contraction_labels!(labels1, labels2, is1, is2, label)
  N1 = length(is1)
  N2 = length(is2)
  @inbounds for n1 in 1:N1, n2 in 1:N2
    i1 = is1[n1]
    i2 = is2[n2]
    if i1 == i2
      labels1[n1] = labels2[n2] = label
      label += 1
    end
  end
  return label
end

function uncommon_contraction_labels!(labels, is, label)
  N = length(labels)
  @inbounds for n in 1:N
    if iszero(labels[n])
      labels[n] = label
      label += 1
    end
  end
  return label
end

function contraction_labels!(labels, is)
  ntensors = length(is)
  nextlabel = 1
  # Loop through each tensor pair searching for
  # common indices
  @inbounds for n1 in 1:(ntensors - 1), n2 in (n1 + 1):ntensors
    nextlabel = common_contraction_labels!(
      labels[n1], labels[n2], is[n1], is[n2], nextlabel
    )
  end
  @inbounds for n in 1:ntensors
    nextlabel = uncommon_contraction_labels!(labels[n], is[n], nextlabel)
  end
  return nextlabel - 1
end

function empty_labels(is::NTuple{N}) where {N}
  return ntuple(n -> fill(0, length(is[n])), Val(N))
end

function empty_labels(is::Vector)
  ntensors = length(is)
  labels = Vector{Vector{Int}}(undef, ntensors)
  @inbounds for n in 1:ntensors
    labels[n] = fill(0, length(is[n]))
  end
  return labels
end

function contraction_labels(is)
  labels = empty_labels(is)
  contraction_labels!(labels, is)
  return labels
end

contraction_labels(is...) = contraction_labels(is)

#
# Use a Dict as a cache to map the indices to the integer label
# This only helps with many nodes/tensors (nnodes > 30)
# TODO: determine the crossover when this is useful and use
# it in `depth_first_constructive`/`breadth_first_constructive`
#

contraction_labels_caching(is) = contraction_labels_caching(eltype(eltype(is)), is)

function contraction_labels_caching(::Type{IndexT}, is) where {IndexT}
  labels = empty_labels(is)
  return contraction_labels_caching!(labels, IndexT, is)
end

function contraction_labels_caching!(labels, ::Type{IndexT}, is) where {IndexT}
  N = length(is)
  ind_to_label = Dict{IndexT,Int}()
  label = 0
  @inbounds for n in 1:N
    isₙ = is[n]
    labelsₙ = labels[n]
    @inbounds for j in 1:length(labelsₙ)
      i = isₙ[j]
      i_label = get!(ind_to_label, i) do
        label += 1
      end
      labelsₙ[j] = i_label
    end
  end
  return label
end

#
# Compute the labels and also return a data structure storing the dims.
#

function label_dims(::Type{DimT}, is) where {DimT<:Integer}
  labels = empty_labels(is)
  nlabels = contraction_labels!(labels, is)
  dims = fill(zero(DimT), nlabels)
  @inbounds for i in 1:length(is)
    labelsᵢ = labels[i]
    isᵢ = is[i]
    @inbounds for n in 1:length(labelsᵢ)
      lₙ = labelsᵢ[n]
      if iszero(dims[lₙ])
        dims[lₙ] = dim(isᵢ[n])
      end
    end
  end
  return labels, dims
end

label_dims(is...) = label_dims(is)

# Convert a contraction sequence in pair form to tree format.
# This is used in `depth_first_constructive` to convert the output.
function pair_sequence_to_tree(pairs::Vector{Pair{Int,Int}}, N::Int)
  trees = Any[1:N...]
  for p in pairs
    push!(trees, Any[trees[p[1]], trees[p[2]]])
  end
  return trees[end]
end

#
# BitSet utilities
#

function _cmp(A::BitSet, B::BitSet)
  for (a, b) in zip(A, B)
    if !isequal(a, b)
      return isless(a, b) ? -1 : 1
    end
  end
  return cmp(length(A), length(B))
end

# Returns true when `A` is less than `B` in lexicographic order.
_isless(A::BitSet, B::BitSet) = _cmp(A, B) < 0

bitset(::Type{BitSet}, ints) = BitSet(ints)

function bitset(::Type{T}, ints) where {T<:Unsigned}
  set = zero(T)
  u = one(T)
  for i in ints
    set |= (u << (i - 1))
  end
  return set
end

# Return a vector of the positions of the nonzero bits
# Used for debugging
function findall_nonzero_bits(i::Unsigned)
  nonzeros = Int[]
  n = 1
  @inbounds while !iszero(i)
    if isodd(i)
      push!(nonzeros, n)
    end
    i = i >> 1
    n += 1
  end
  return nonzeros
end

# Return the position of the first nonzero bit
function findfirst_nonzero_bit(i::Unsigned)
  n = 0
  @inbounds while !iszero(i)
    if isodd(i)
      return n + 1
    end
    i = i >> 1
    n += 1
  end
  return n
end

_isless(s1::T, s2::T) where {T<:Unsigned} = s1 < s2
_intersect(s1::BitSet, s2::BitSet) = intersect(s1, s2)
_intersect(s1::T, s2::T) where {T<:Unsigned} = s1 & s2
_union(s1::BitSet, s2::BitSet) = union(s1, s2)
_union(s1::T, s2::T) where {T<:Unsigned} = s1 | s2
_setdiff(s1::BitSet, s2::BitSet) = setdiff(s1, s2)
_setdiff(s1::T, s2::T) where {T<:Unsigned} = s1 & (~s2)
_symdiff(s1::BitSet, s2::BitSet) = symdiff(s1, s2)
_symdiff(s1::T, s2::T) where {T<:Unsigned} = xor(s1, s2)
_isemptyset(s::BitSet) = isempty(s)
_isemptyset(s::Unsigned) = iszero(s)

# TODO: use _first instead, optimize to avoid using _set
_only(s::BitSet) = only(s)
_only(s::Unsigned) = findfirst_nonzero_bit(s)

#
# Adjacency matrix and connected components
#

# For a network of tensors T (stored as index labels), return the adjacency matrix.
function adjacencymatrix(T::Vector, alldims::Vector)
  # First break up the network into disconnected parts
  N = length(T)
  _adjacencymatrix = falses(N, N)
  for nᵢ in 1:(N - 1), nⱼ in (nᵢ + 1):N
    if dim(_intersect(T[nᵢ], T[nⱼ]), alldims) > 1
      _adjacencymatrix[nᵢ, nⱼ] = _adjacencymatrix[nⱼ, nᵢ] = true
    end
  end
  return _adjacencymatrix
end

# For a given adjacency matrix of size n x n, connectedcomponents returns
# a list of components that contains integer vectors, where every integer
# vector groups the indices of the vertices of a connected component of the
# graph encoded by A. The number of connected components is given by
# length(components).
function connectedcomponents(A::AbstractMatrix{Bool})
  n = size(A, 1)
  @assert size(A, 2) == n
  components = Vector{Vector{Int}}(undef, 0)
  assignedlist = falses((n,))
  for i in 1:n
    if !assignedlist[i]
      assignedlist[i] = true
      checklist = [i]
      currentcomponent = [i]
      while !isempty(checklist)
        j = pop!(checklist)
        for k in findall(A[j, :])
          if !assignedlist[k]
            push!(currentcomponent, k)
            push!(checklist, k)
            assignedlist[k] = true
          end
        end
      end
      push!(components, currentcomponent)
    end
  end
  return components
end

# For a network of tensors T (stored as index labels), return the connected components
# (splits up T into the connected components).
function connectedcomponents(T::Vector, alldims::Vector)
  return connectedcomponents(adjacencymatrix(T, alldims))
end
