function LinearAlgebra.eigen(a::BlockSparseArray)
  return error("Not implemented")
end

# TODO: Maybe make `Hermitian` partially eager for `BlockSparseArray`?
function LinearAlgebra.eigen(
  a::Union{Hermitian{<:Real,<:BlockSparseArray},Hermitian{<:Complex,<:BlockSparseArray}}
)
  # TODO: Test `a` is block diagonal.
  # @assert is_block_diagonal(a)
  d = BlockSparseArray{real(eltype(a))}(axes(a, 1))
  u = BlockSparseArray{eltype(a)}(axes(a))
  for b in nonzero_blockkeys(a)
    d_b, u_b = eigen(@view a[b])
    d[BlockArrays.Block(b.n[1])] = d_b
    u[b] = u_b
  end
  return d, u
end
