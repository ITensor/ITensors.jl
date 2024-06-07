using BlockArrays: BlockedUnitRange

# Represents the range `1:1` or `Base.OneTo(1)`.
struct OneToOne{T} <: AbstractUnitRange{T} end
OneToOne() = OneToOne{Bool}()
Base.first(a::OneToOne) = one(eltype(a))
Base.last(a::OneToOne) = one(eltype(a))

gradedisequal(::AbstractUnitRange, ::OneToOne) = false
gradedisequal(::OneToOne, ::AbstractUnitRange) = false
gradedisequal(::OneToOne, ::OneToOne) = true

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

function tensor_product(::AbstractUnitRange, ::AbstractUnitRange)
  return error("Not implemented yet.")
end

function tensor_product(a1::Base.OneTo, a2::Base.OneTo)
  return Base.OneTo(length(a1) * length(a2))
end

function tensor_product(::OneToOne, a2::AbstractUnitRange)
  return a2
end

function tensor_product(a1::AbstractUnitRange, ::OneToOne)
  return a1
end

function tensor_product(::OneToOne, ::OneToOne)
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

flatten_maybe_nested(v::Vector{<:Integer}) = v
flatten_maybe_nested(v::Vector{<:GradedUnitRange}) = reduce(vcat, blocklengths.(v))

using BlockArrays: blockedrange, blocks
function tensor_product(a1::BlockedUnitRange, a2::BlockedUnitRange)
  maybe_nested = map(
    it -> mapreduce(length, fuse_blocklengths, it),
    Iterators.flatten((Iterators.product(blocks(a1), blocks(a2)),)),
  )
  blocklengths = flatten_maybe_nested(maybe_nested)
  return blockedrange(blocklengths)
end

function blocksortperm(a::BlockedUnitRange)
  return Block.(sortperm(blocklabels(a)))
end

# convention: sort UnitRangeDual according to nondual blocks
function blocksortperm(a::UnitRangeDual)
  return Block.(sortperm(blocklabels(nondual(a))))
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
function blockmergesortperm(a::BlockedUnitRange)
  return Block.(groupsortperm(blocklabels(a)))
end

# Used by `TensorAlgebra.splitdims` in `BlockSparseArraysGradedAxesExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

function blockmergesortperm(a::UnitRangeDual)
  return Block.(groupsortperm(blocklabels(nondual(a))))
end

function blockmergesort(g::GradedUnitRange)
  glabels = blocklabels(g)
  gblocklengths = blocklengths(g)
  new_blocklengths = map(
    la -> labelled(sum(gblocklengths[findall(==(la), glabels)]; init=0), la),
    sort(unique(glabels)),
  )
  return GradedAxes.gradedrange(new_blocklengths)
end

blockmergesort(g::UnitRangeDual) = dual(blockmergesort(nondual(g)))
blockmergesort(g::OneToOne) = g

# fusion_product produces a sorted, non-dual GradedUnitRange
function fusion_product(g1, g2)
  return blockmergesort(tensor_product(g1, g2))
end

fusion_product(g::AbstractUnitRange) = blockmergesort(g)
fusion_product(g::UnitRangeDual) = fusion_product(label_dual(nondual(g)))

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
