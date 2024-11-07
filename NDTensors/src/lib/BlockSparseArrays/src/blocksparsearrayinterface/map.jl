function map_stored_blocks(f, a::AbstractArray)
  # TODO: Implement this as:
  # ```julia
  # mapped_blocks = SparseArraysInterface.map_stored(f, blocks(a))
  # BlockSparseArray(mapped_blocks, axes(a))
  # ```
  # TODO: `block_stored_indices` should output `Indices` storing
  # the stored Blocks, not a `Dictionary` from cartesian indices
  # to Blocks.
  bs = collect(block_stored_indices(a))
  ds = map(b -> f(@view(a[b])), bs)
  # We manually specify the block type using `Base.promote_op`
  # since `a[b]` may not be inferrable. For example, if `blocktype(a)`
  # is `Diagonal{Float64,Vector{Float64}}`, the non-stored blocks are `Matrix{Float64}`
  # since they can't necessarily by `Diagonal` if there are rectangular blocks.
  mapped_blocks = Dictionary{eltype(bs),eltype(ds)}(bs, ds)
  # TODO: Use `similartype(typeof(a), eltype(eltype(mapped_blocks)))(...)`.
  return BlockSparseArray(mapped_blocks, axes(a))
end
