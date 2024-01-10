using BlockArrays: AbstractBlockArray, Block, blockedrange
using Dictionaries: Dictionary, Indices
using ..SparseArrayInterface: stored_indices

# TODO: Use `Tuple` conversion once
# BlockArrays.jl PR is merged.
block_to_cartesianindex(b::Block) = CartesianIndex(b.n)

function blocks_to_cartesianindices(i::Indices{<:Block})
  return block_to_cartesianindex.(i)
end

function blocks_to_cartesianindices(d::Dictionary{<:Block})
  return Dictionary(blocks_to_cartesianindices(eachindex(d)), d)
end

function block_reshape(a::AbstractBlockArray, dims::Tuple{Vararg{Vector{Int}}})
  return block_reshape(a, blockedrange.(dims))
end

function block_reshape(a::AbstractBlockArray, dims::Vararg{Vector{Int}})
  return block_reshape(a, dims)
end

tuple_oneto(n) = ntuple(identity, n)

function block_reshape(a::AbstractBlockArray, axes::Tuple{Vararg{AbstractUnitRange}})
  reshaped_blocks_a = reshape(blocks(a), blocklength.(axes))
  reshaped_a = similar(a, axes)
  for I in stored_indices(reshaped_blocks_a)
    block_size_I = map(i -> length(axes[i][Block(I[i])]), tuple_oneto(length(axes)))
    # TODO: Better converter here.
    reshaped_a[Block(Tuple(I))] = reshape(reshaped_blocks_a[I], block_size_I)
  end
  return reshaped_a
end

function block_reshape(a::AbstractBlockArray, axes::Vararg{AbstractUnitRange})
  return block_reshape(a, axes)
end
