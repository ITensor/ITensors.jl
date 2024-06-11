using BlockArrays: AbstractBlockedUnitRange

# Represents the range `1:1` or `Base.OneTo(1)`.
struct OneToOne{T} <: AbstractUnitRange{T} end
OneToOne() = OneToOne{Bool}()
Base.first(a::OneToOne) = one(eltype(a))
Base.last(a::OneToOne) = one(eltype(a))

# https://github.com/ITensor/ITensors.jl/blob/v0.3.57/NDTensors/src/lib/GradedAxes/src/tensor_product.jl
# https://en.wikipedia.org/wiki/Tensor_product
# https://github.com/KeitaNakamura/Tensorial.jl
function tensor_product(
  a1::AbstractUnitRange,
  a2::AbstractUnitRange,
  a3::AbstractUnitRange,
  a_rest::Vararg{AbstractUnitRange},
)
  return foldl(tensor_product, (a1, a2, a3, a_rest...))
end

function tensor_product(a1::AbstractUnitRange, a2::AbstractUnitRange)
  return error("Not implemented yet.")
end

function tensor_product(a1::Base.OneTo, a2::Base.OneTo)
  return Base.OneTo(length(a1) * length(a2))
end

function tensor_product(a1::OneToOne, a2::AbstractUnitRange)
  return a2
end

function tensor_product(a1::AbstractUnitRange, a2::OneToOne)
  return a1
end

function tensor_product(a1::OneToOne, a2::OneToOne)
  return OneToOne()
end

function fuse_labels(x, y)
  return error(
    "`fuse_labels` not implemented for object of type `$(typeof(x))` and `$(typeof(y))`."
  )
end

function fuse_blocklengths(x::Integer, y::Integer)
  return x * y
end

using ..LabelledNumbers: LabelledInteger, label, labelled, unlabel
function fuse_blocklengths(x::LabelledInteger, y::LabelledInteger)
  return labelled(unlabel(x) * unlabel(y), fuse_labels(label(x), label(y)))
end

using BlockArrays: blockedrange, blocks
function tensor_product(a1::AbstractBlockedUnitRange, a2::AbstractBlockedUnitRange)
  blocklengths = map(vec(collect(Iterators.product(blocks(a1), blocks(a2))))) do x
    return mapreduce(length, fuse_blocklengths, x)
  end
  return blockedrange(blocklengths)
end

function blocksortperm(a::AbstractBlockedUnitRange)
  # TODO: Figure out how to deal with dual sectors.
  # TODO: `rev=isdual(a)`  may not be correct for symmetries beyond `U(1)`.
  ## return Block.(sortperm(nondual_sectors(a); rev=isdual(a)))
  return Block.(sortperm(blocklabels(a)))
end

using BlockArrays: Block, BlockVector
using SplitApplyCombine: groupcount
# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function groupsortperm(v; kwargs...)
  perm = sortperm(v; kwargs...)
  v_sorted = @view v[perm]
  group_lengths = collect(groupcount(identity, v_sorted))
  return BlockVector(perm, group_lengths)
end

# Used by `TensorAlgebra.splitdims` in `BlockSparseArraysGradedAxesExt`.
# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function blockmergesortperm(a::AbstractBlockedUnitRange)
  # If it is dual, reverse the sorting so the sectors
  # end up sorted in the same way whether or not the space
  # is dual.
  # TODO: Figure out how to deal with dual sectors.
  # TODO: `rev=isdual(a)`  may not be correct for symmetries beyond `U(1)`.
  ## return Block.(groupsortperm(nondual_sectors(a); rev=isdual(a)))
  return Block.(groupsortperm(blocklabels(a)))
end

# Used by `TensorAlgebra.splitdims` in `BlockSparseArraysGradedAxesExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

# Used by `TensorAlgebra.fusedims` in `BlockSparseArraysGradedAxesExt`.
function blockmergesortperm(a::GradedUnitRange)
  # If it is dual, reverse the sorting so the sectors
  # end up sorted in the same way whether or not the space
  # is dual.
  # TODO: Figure out how to deal with dual sectors.
  # TODO: `rev=isdual(a)`  may not be correct for symmetries beyond `U(1)`.
  return Block.(groupsortperm(blocklabels(a)))
end
