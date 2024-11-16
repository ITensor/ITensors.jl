using ArrayLayouts: ArrayLayouts, DualLayout, MemoryLayout, MulAdd
using BlockArrays: BlockLayout
using ..SparseArraysBase: SparseLayout
using ..TypeParameterAccessors: parenttype, similartype

function ArrayLayouts.MemoryLayout(arraytype::Type{<:AnyAbstractBlockSparseArray})
  outer_layout = typeof(MemoryLayout(blockstype(arraytype)))
  inner_layout = typeof(MemoryLayout(blocktype(arraytype)))
  return BlockLayout{outer_layout,inner_layout}()
end

# TODO: Generalize to `BlockSparseVectorLike`/`AnyBlockSparseVector`.
function ArrayLayouts.MemoryLayout(
  arraytype::Type{<:Adjoint{<:Any,<:AbstractBlockSparseVector}}
)
  return DualLayout{typeof(MemoryLayout(parenttype(arraytype)))}()
end
# TODO: Generalize to `BlockSparseVectorLike`/`AnyBlockSparseVector`.
function ArrayLayouts.MemoryLayout(
  arraytype::Type{<:Transpose{<:Any,<:AbstractBlockSparseVector}}
)
  return DualLayout{typeof(MemoryLayout(parenttype(arraytype)))}()
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
  # TODO: Define `blocktype`/`blockstype` for `SubArray` wrapping `BlockSparseArray`.
  # TODO: Use `similar`?
  blocktype_a = blocktype(parent(a))
  a_dest = BlockSparseArray{eltype(a),length(axes),blocktype_a}(axes)
  a_dest .= a
  return a_dest
end

# Materialize a SubArray view.
function ArrayLayouts.sub_materialize(
  layout::BlockLayout{<:SparseLayout}, a, axes::Tuple{Vararg{Base.OneTo}}
)
  a_dest = blocktype(a)(undef, length.(axes))
  a_dest .= a
  return a_dest
end
