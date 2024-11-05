function map_stored_blocks(f, a::AbstractArray)
  # TODO: Implement this as:
  # ```julia
  # mapped_blocks = SparseArraysInterface.map_stored(f, blocks(a))
  # BlockSparseArray(mapped_blocks, axes(a))
  # ```
  # TODO: `block_stored_indices` should output `Indices` storing
  # the stored Blocks, not a `Dictionary` from cartesian indices
  # to Blocks.
  bs = block_stored_indices(a)
  mapped_blocks = Dictionary(bs, map(b -> f(@view(a[b])), bs))
  # TODO: Use `similartype(typeof(a), eltype(eltype(mapped_blocks)))(...)`.
  return BlockSparseArray(mapped_blocks, axes(a))
end
