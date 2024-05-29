using BlockArrays: BlockedUnitRange

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

# Handle dual. Always return a non-dual GradedUnitRange.
function tensor_product(a1::AbstractUnitRange, a2::UnitRangeDual)
  return tensor_product(a1, label_dual(dual(a2)))
end

function tensor_product(a1::UnitRangeDual, a2::AbstractUnitRange)
  return tensor_product(label_dual(dual(a1)), a2)
end

function tensor_product(a1::UnitRangeDual, a2::UnitRangeDual)
  return tensor_product(label_dual(dual(a1)), label_dual(dual(a2)))
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
function tensor_product(a1::BlockedUnitRange, a2::BlockedUnitRange)
  blocklengths = map(vec(collect(Iterators.product(blocks(a1), blocks(a2))))) do x
    return mapreduce(length, fuse_blocklengths, x)
  end
  return blockedrange(blocklengths)
end

function blocksortperm(a::BlockedUnitRange)
  return Block.(sortperm(blocklabels(a)))
end

function blocksortperm(a::UnitRangeDual)
  # If it is dual, reverse the sorting so the sectors
  # end up sorted in the same way whether or not the space
  # is dual.
  return Block.(sortperm(blocklabels(label_dual(dual(a)))))
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
function blockmergesortperm(a::BlockedUnitRange)
  return Block.(groupsortperm(blocklabels(a)))
end

# Used by `TensorAlgebra.splitdims` in `BlockSparseArraysGradedAxesExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

function blockmergesortperm(a::UnitRangeDual)
  # If it is dual, reverse the sorting so the sectors
  # end up sorted in the same way whether or not the space
  # is dual.
  return Block.(groupsortperm(blocklabels(label_dual(dual(a)))))
end

# fusion_product generalizes tensor_product to non-abelian groups and fusion categories
# in the case of abelian groups, it is equivalent to tensor_product + applying blockmergesortperm
function fusion_product(::AbstractUnitRange, ::AbstractUnitRange)
  return error("Not implemented")
end

# recursive fusion_product. Simpler than reduce + fix type stability issues with reduce
function fusion_product(g1, g2, g3...)
  return fusion_product(fusion_product(g1, g2), g3...)
end

# Handle dual. Always return a non-dual GradedUnitRange.
function fusion_product(g1::UnitRangeDual, g2::AbstractUnitRange)
  return fusion_product(label_dual(dual(g1)), g2)
end
function fusion_product(g1::AbstractUnitRange, g2::UnitRangeDual)
  return fusion_product(g1, label_dual(dual(g2)))
end
function fusion_product(g1::UnitRangeDual, g2::UnitRangeDual)
  return fusion_product(label_dual(dual(g1)), label_dual(dual(g2)))
end
