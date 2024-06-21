using BlockArrays:
  AbstractBlockVector,
  Block,
  BlockedUnitRange,
  BlockIndex,
  block,
  blockcheckbounds,
  blocks,
  blocklengths,
  findblockindex
using LinearAlgebra: Adjoint, Transpose
using ..SparseArrayInterface: perm, iperm, nstored, sparse_zero!
## using MappedArrays: mappedarray

blocksparse_blocks(a::AbstractArray) = error("Not implemented")

function blocksparse_getindex(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  return a[findblockindex.(axes(a), I)...]
end

function blocksparse_getindex(a::AbstractArray{<:Any,N}, I::Block{N}) where {N}
  return blocksparse_getindex(a, Tuple(I)...)
end
function blocksparse_getindex(a::AbstractArray{<:Any,N}, I::Vararg{Block{1},N}) where {N}
  # TODO: Avoid copy if the block isn't stored.
  return copy(blocks(a)[Int.(I)...])
end

# TODO: Implement as `copy(@view a[I...])`, which is then implemented
# through `ArrayLayouts.sub_materialize`.
using ..SparseArrayInterface: set_getindex_zero_function
function blocksparse_getindex(
  a::AbstractArray{<:Any,N}, I::Vararg{AbstractVector{<:Block{1}},N}
) where {N}
  blocks_a = blocks(a)
  # Convert to cartesian indices of the underlying sparse array of blocks.
  CI = map(i -> Int.(i), I)
  subblocks_a = blocks_a[CI...]
  subaxes = ntuple(ndims(a)) do i
    return only(axes(axes(a, i)[I[i]]))
  end
  subblocks_a = set_getindex_zero_function(subblocks_a, BlockZero(subaxes))
  return typeof(a)(subblocks_a, subaxes)
end

# Slice by block and merge according to the blocking structure of the indices.
function blocksparse_getindex(
  a::AbstractArray{<:Any,N}, I::Vararg{AbstractBlockVector{<:Block{1}},N}
) where {N}
  a_sub = a[Vector.(I)...]
  # TODO: Define `blocklengths(::AbstractBlockVector)`? Maybe make a PR
  # to `BlockArrays`.
  blockmergers = blockedrange.(blocklengths.(only.(axes.(I))))
  # TODO: Need to implement this!
  a_merged = block_merge(a_sub, blockmergers...)
  return a_merged
end

# a[1:2, 1:2]
# TODO: This definition means that the result of slicing a block sparse array
# with a non-blocked unit range is blocked. We may want to change that behavior,
# and make that explicit with `@blocked a[1:2, 1:2]`. See the discussion in
# https://github.com/JuliaArrays/BlockArrays.jl/issues/347 and also
# https://github.com/ITensor/ITensors.jl/issues/1336.
function blocksparse_to_indices(a, inds, I::Tuple{UnitRange{<:Integer},Vararg{Any}})
  bs1 = blockrange(inds[1], I[1])
  I1 = BlockSlice(bs1, blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# a[[Block(2), Block(1)], [Block(2), Block(1)]]
function blocksparse_to_indices(a, inds, I::Tuple{Vector{<:Block{1}},Vararg{Any}})
  I1 = BlockIndices(I[1], blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# a[BlockVector([Block(2), Block(1)], [2]), BlockVector([Block(2), Block(1)], [2])]
# Permute and merge blocks.
# TODO: This isn't merging blocks yet, that needs to be implemented that.
function blocksparse_to_indices(
  a::BlockSparseArrayLike, inds, I::Tuple{BlockVector{<:Block{1}},Vararg{Any}}
)
  I1 = BlockIndices(I[1], blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# TODO: Need to implement this!
function block_merge end

function blocksparse_setindex!(a::AbstractArray{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  a[findblockindex.(axes(a), I)...] = value
  return a
end

function blocksparse_setindex!(a::AbstractArray{<:Any,N}, value, I::BlockIndex{N}) where {N}
  i = Int.(Tuple(block(I)))
  a_b = blocks(a)[i...]
  a_b[I.α...] = value
  # Set the block, required if it is structurally zero.
  blocks(a)[i...] = a_b
  return a
end

function blocksparse_setindex!(a::AbstractArray{<:Any,N}, value, I::Block{N}) where {N}
  blocksparse_setindex!(a, value, Tuple(I)...)
  return a
end
function blocksparse_setindex!(
  a::AbstractArray{<:Any,N}, value, I::Vararg{Block{1},N}
) where {N}
  i = Int.(I)
  @boundscheck blockcheckbounds(a, i...)
  # TODO: Use `blocksizes(a)[i...]` when we upgrade to
  # BlockArrays.jl v1.
  if size(value) ≠ size(view(a, I...))
    return throw(
      DimensionMismatch("Trying to set a block with an array of the wrong size.")
    )
  end
  blocks(a)[i...] = value
  return a
end

function blocksparse_fill!(a::AbstractArray, value)
  for b in BlockRange(a)
    # We can't use:
    # ```julia
    # a[b] .= value
    # ```
    # since that would lead to a stack overflow,
    # because broadcasting calls `fill!`.

    # TODO: Ideally we would use:
    # ```julia
    # @view!(a[b]) .= value
    # ```
    # but that doesn't work on `SubArray` right now.

    # This line is needed to instantiate blocks
    # that aren't instantiated yet. Maybe
    # we can make this work without this line?
    blocks(a)[Int.(Tuple(b))...] = blocks(a)[Int.(Tuple(b))...]
    blocks(a)[Int.(Tuple(b))...] .= value
  end
  return a
end

function block_nstored(a::AbstractArray)
  return nstored(blocks(a))
end

# BlockArrays

using ..SparseArrayInterface:
  SparseArrayInterface, AbstractSparseArray, AbstractSparseMatrix

_perm(::PermutedDimsArray{<:Any,<:Any,P}) where {P} = P
_getindices(t::Tuple, indices) = map(i -> t[i], indices)
_getindices(i::CartesianIndex, indices) = CartesianIndex(_getindices(Tuple(i), indices))

# Represents the array of arrays of a `PermutedDimsArray`
# wrapping a block spare array, i.e. `blocks(array)` where `a` is a `PermutedDimsArray`.
struct SparsePermutedDimsArrayBlocks{T,N,Array<:PermutedDimsArray{T,N}} <:
       AbstractSparseArray{T,N}
  array::Array
end
function blocksparse_blocks(a::PermutedDimsArray)
  return SparsePermutedDimsArrayBlocks(a)
end
function Base.size(a::SparsePermutedDimsArrayBlocks)
  return _getindices(size(blocks(parent(a.array))), _perm(a.array))
end
function Base.getindex(
  a::SparsePermutedDimsArrayBlocks{<:Any,N}, index::Vararg{Int,N}
) where {N}
  return PermutedDimsArray(
    blocks(parent(a.array))[_getindices(index, _perm(a.array))...], _perm(a.array)
  )
end
function SparseArrayInterface.stored_indices(a::SparsePermutedDimsArrayBlocks)
  return map(I -> _getindices(I, _perm(a.array)), stored_indices(blocks(parent(a.array))))
end
# TODO: Either make this the generic interface or define
# `SparseArrayInterface.sparse_storage`, which is used
# to defined this.
SparseArrayInterface.nstored(a::SparsePermutedDimsArrayBlocks) = length(stored_indices(a))
function SparseArrayInterface.sparse_storage(a::SparsePermutedDimsArrayBlocks)
  return error("Not implemented")
end

reverse_index(index) = reverse(index)
reverse_index(index::CartesianIndex) = CartesianIndex(reverse(Tuple(index)))

# Represents the array of arrays of a `Transpose`
# wrapping a block spare array, i.e. `blocks(array)` where `a` is a `Transpose`.
struct SparseTransposeBlocks{T,Array<:Transpose{T}} <: AbstractSparseMatrix{T}
  array::Array
end
function blocksparse_blocks(a::Transpose)
  return SparseTransposeBlocks(a)
end
function Base.size(a::SparseTransposeBlocks)
  return reverse(size(blocks(parent(a.array))))
end
function Base.getindex(a::SparseTransposeBlocks, index::Vararg{Int,2})
  return transpose(blocks(parent(a.array))[reverse(index)...])
end
# TODO: This should be handled by generic `AbstractSparseArray` code.
function Base.getindex(a::SparseTransposeBlocks, index::CartesianIndex{2})
  return a[Tuple(index)...]
end
# TODO: Create a generic `parent_index` function to map an index
# a parent index.
function Base.isassigned(a::SparseTransposeBlocks, index::Vararg{Int,2})
  return isassigned(blocks(parent(a.array)), reverse(index)...)
end
function SparseArrayInterface.stored_indices(a::SparseTransposeBlocks)
  return map(reverse_index, stored_indices(blocks(parent(a.array))))
end
# TODO: Either make this the generic interface or define
# `SparseArrayInterface.sparse_storage`, which is used
# to defined this.
SparseArrayInterface.nstored(a::SparseTransposeBlocks) = length(stored_indices(a))
function SparseArrayInterface.sparse_storage(a::SparseTransposeBlocks)
  return error("Not implemented")
end

# Represents the array of arrays of a `Adjoint`
# wrapping a block spare array, i.e. `blocks(array)` where `a` is a `Adjoint`.
struct SparseAdjointBlocks{T,Array<:Adjoint{T}} <: AbstractSparseMatrix{T}
  array::Array
end
function blocksparse_blocks(a::Adjoint)
  return SparseAdjointBlocks(a)
end
function Base.size(a::SparseAdjointBlocks)
  return reverse(size(blocks(parent(a.array))))
end
# TODO: Create a generic `parent_index` function to map an index
# a parent index.
function Base.getindex(a::SparseAdjointBlocks, index::Vararg{Int,2})
  return blocks(parent(a.array))[reverse(index)...]'
end
# TODO: Create a generic `parent_index` function to map an index
# a parent index.
# TODO: This should be handled by generic `AbstractSparseArray` code.
function Base.getindex(a::SparseAdjointBlocks, index::CartesianIndex{2})
  return a[Tuple(index)...]
end
# TODO: Create a generic `parent_index` function to map an index
# a parent index.
function Base.isassigned(a::SparseAdjointBlocks, index::Vararg{Int,2})
  return isassigned(blocks(parent(a.array)), reverse(index)...)
end
function SparseArrayInterface.stored_indices(a::SparseAdjointBlocks)
  return map(reverse_index, stored_indices(blocks(parent(a.array))))
end
# TODO: Either make this the generic interface or define
# `SparseArrayInterface.sparse_storage`, which is used
# to defined this.
SparseArrayInterface.nstored(a::SparseAdjointBlocks) = length(stored_indices(a))
function SparseArrayInterface.sparse_storage(a::SparseAdjointBlocks)
  return error("Not implemented")
end

# Represents the array of arrays of a `SubArray`
# wrapping a block spare array, i.e. `blocks(array)` where `a` is a `SubArray`.
struct SparseSubArrayBlocks{T,N,Array<:SubArray{T,N}} <: AbstractSparseArray{T,N}
  array::Array
end
# TODO: Define this as `blockrange(a::AbstractArray, indices::Tuple{Vararg{AbstractUnitRange}})`.
function blockrange(a::SparseSubArrayBlocks)
  blockranges = blockrange.(axes(parent(a.array)), a.array.indices)
  return map(blockrange -> Int.(blockrange), blockranges)
end
function Base.axes(a::SparseSubArrayBlocks)
  return Base.OneTo.(length.(blockrange(a)))
end
function Base.size(a::SparseSubArrayBlocks)
  return length.(axes(a))
end
function Base.getindex(a::SparseSubArrayBlocks{<:Any,N}, I::Vararg{Int,N}) where {N}
  parent_blocks = @view blocks(parent(a.array))[blockrange(a)...]
  parent_block = parent_blocks[I...]
  # TODO: Define this using `blockrange(a::AbstractArray, indices::Tuple{Vararg{AbstractUnitRange}})`.
  block = Block(ntuple(i -> blockrange(a)[i][I[i]], ndims(a)))
  return @view parent_block[blockindices(parent(a.array), block, a.array.indices)...]
end
# TODO: This should be handled by generic `AbstractSparseArray` code.
function Base.getindex(a::SparseSubArrayBlocks{<:Any,N}, I::CartesianIndex{N}) where {N}
  return a[Tuple(I)...]
end
function Base.setindex!(a::SparseSubArrayBlocks{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  parent_blocks = @view blocks(parent(a.array))[blockrange(a)...]
  # TODO: The following line is required to instantiate
  # uninstantiated blocks, maybe use `@view!` instead,
  # or some other code pattern.
  parent_blocks[I...] = parent_blocks[I...]
  # TODO: Define this using `blockrange(a::AbstractArray, indices::Tuple{Vararg{AbstractUnitRange}})`.
  block = Block(ntuple(i -> blockrange(a)[i][I[i]], ndims(a)))
  return parent_blocks[I...][blockindices(parent(a.array), block, a.array.indices)...] =
    value
end
function Base.isassigned(a::SparseSubArrayBlocks{<:Any,N}, I::Vararg{Int,N}) where {N}
  if CartesianIndex(I) ∉ CartesianIndices(a)
    return false
  end
  # TODO: Implement this properly.
  return true
end
function SparseArrayInterface.stored_indices(a::SparseSubArrayBlocks)
  return stored_indices(view(blocks(parent(a.array)), blockrange(a)...))
end
# TODO: Either make this the generic interface or define
# `SparseArrayInterface.sparse_storage`, which is used
# to defined this.
SparseArrayInterface.nstored(a::SparseSubArrayBlocks) = length(stored_indices(a))
function SparseArrayInterface.sparse_storage(a::SparseSubArrayBlocks)
  return error("Not implemented")
end

function blocksparse_blocks(a::SubArray)
  return SparseSubArrayBlocks(a)
end

using BlockArrays: BlocksView
# TODO: Is this correct in general?
SparseArrayInterface.nstored(a::BlocksView) = 1
