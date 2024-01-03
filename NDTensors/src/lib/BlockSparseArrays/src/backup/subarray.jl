function Base.axes(
  a::SubArray{<:Any,<:Any,<:BlockSparseArray,<:Tuple{Vararg{<:Vector{<:Block}}}}
)
  axes_src = axes(parent(a))
  axes_parent = parentindices(a)
  return sub_axes(axes_src, axes_parent)
end

# Map the location in the sliced array of a parent block `b` to the location
# in a block slice. Return `nothing` if the block
# isn't in the slice.
function sub_blockkey(
  a::SubArray{<:Any,<:Any,<:BlockSparseArray,<:Tuple{Vararg{<:Vector{<:Block}}}}, b::Block
)
  btuple = blocktuple(b)
  dims_a = ntuple(identity, ndims(a))
  subblock_ranges = parentindices(a)
  I = map(i -> findfirst(==(btuple[i]), subblock_ranges[i]), dims_a)
  if any(isnothing, I)
    return nothing
  end
  return Block(I)
end

function nonzero_sub_blockkeys(
  a::SubArray{<:Any,<:Any,<:BlockSparseArray,<:Tuple{Vararg{<:Vector{<:Block}}}}
)
  sub_blockkeys_a = Block{ndims(a),Int}[]
  parent_blockkeys_a = Block{ndims(a),Int}[]
  for b in nonzero_blockkeys(parent(a))
    sub_b = sub_blockkey(a, b)
    if !isnothing(sub_b)
      push!(parent_blockkeys_a, b)
      push!(sub_blockkeys_a, sub_b)
    end
  end
  return Dictionary(parent_blockkeys_a, sub_blockkeys_a)
end

# TODO: Look at what `BlockArrays.jl` does for block slicing, i.e.
# ArrayLayouts.sub_materialize
# ArrayLayouts.MemoryLayout, BlockLayout, BlockSparseLayout
# etc.
function Base.copy(
  a::SubArray{<:Any,<:Any,<:BlockSparseArray,<:Tuple{Vararg{<:Vector{<:Block}}}}
)
  axes_dest = axes(a)
  nonzero_sub_blockkeys_a = nonzero_sub_blockkeys(a)
  # TODO: Preserve the grades here.
  a_dest = BlockSparseArray{eltype(a)}(blocklengths.(axes_dest)...)
  for b in keys(nonzero_sub_blockkeys_a)
    a_dest[nonzero_sub_blockkeys_a[b]] = parent(a)[b]
  end
  return a_dest
end

# Permute the blocks
function Base.getindex(a::BlockSparseArray, b::Vararg{Vector{<:Block{1}}})
  # Lazy version
  ap = @view a[b...]
  return copy(ap)
end
