using ArrayLayouts: ArrayLayouts, MemoryLayout, MulAdd
using BlockArrays: BlockLayout
using ..SparseArrayInterface: SparseLayout

function ArrayLayouts.MemoryLayout(arraytype::Type{<:BlockSparseArrayLike})
  outer_layout = typeof(MemoryLayout(blockstype(arraytype)))
  inner_layout = typeof(MemoryLayout(blocktype(arraytype)))
  return BlockLayout{outer_layout,inner_layout}()
end

function Base.similar(
  ::MulAdd{<:BlockLayout{<:SparseLayout},<:BlockLayout{<:SparseLayout}}, elt::Type, axes
)
  return similar(BlockSparseArray{elt}, axes)
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
