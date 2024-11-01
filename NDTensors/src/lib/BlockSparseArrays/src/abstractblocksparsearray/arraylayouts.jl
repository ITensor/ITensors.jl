using ArrayLayouts: ArrayLayouts, MemoryLayout, MulAdd
using BlockArrays: BlockLayout
using ..SparseArrayInterface: SparseLayout
# TODO: Move to `NDTensors.TypeParameterAccessors`.
using ..NDTensors: similartype

function ArrayLayouts.MemoryLayout(arraytype::Type{<:BlockSparseArrayLike})
  outer_layout = typeof(MemoryLayout(blockstype(arraytype)))
  inner_layout = typeof(MemoryLayout(blocktype(arraytype)))
  return BlockLayout{outer_layout,inner_layout}()
end

function Base.similar(
  mul::MulAdd{<:BlockLayout{<:SparseLayout},<:BlockLayout{<:SparseLayout},<:Any,<:Any,A,B},
  elt::Type,
  axes,
) where {A,B}
  # TODO: Check that this equals `similartype(blocktype(B), elt, axes)`,
  # or maybe promote them?
  output_blocktype = similartype(blocktype(A), elt, axes)
  return similar(BlockSparseArray{elt,length(axes),output_blocktype}, axes)
end

# Materialize a SubArray view.
function ArrayLayouts.sub_materialize(layout::BlockLayout{<:SparseLayout}, a, axes)
  # TODO: Make more generic for GPU.
  a_dest = BlockSparseArray{eltype(a)}(axes)
  a_dest .= a
  return a_dest
end

# Materialize a SubArray view.
function ArrayLayouts.sub_materialize(
  layout::BlockLayout{<:SparseLayout}, a, axes::Tuple{Vararg{Base.OneTo}}
)
  # TODO: Make more generic for GPU.
  a_dest = Array{eltype(a)}(undef, length.(axes))
  a_dest .= a
  return a_dest
end
