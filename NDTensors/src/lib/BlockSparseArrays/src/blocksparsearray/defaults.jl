using BlockArrays: Block
using Dictionaries: Dictionary
using ..SparseArraysBase: SparseArrayDOK

# Construct the sparse structure storing the blocks
function default_blockdata(
  block_data::Dictionary{<:CartesianIndex{N},<:AbstractArray{<:Any,N}},
  axes::Tuple{Vararg{AbstractUnitRange,N}},
) where {N}
  return error()
end

function default_blocks(
  block_indices::Vector{<:Block{N}},
  block_data::Vector{<:AbstractArray{<:Any,N}},
  axes::Tuple{Vararg{AbstractUnitRange,N}},
) where {N}
  return default_blocks(Dictionary(block_indices, block_data), axes)
end

function default_blocks(
  block_data::Dictionary{<:Block{N},<:AbstractArray{<:Any,N}},
  axes::Tuple{Vararg{AbstractUnitRange,N}},
) where {N}
  return default_blocks(blocks_to_cartesianindices(block_data), axes)
end

function default_arraytype(elt::Type, axes::Tuple{Vararg{AbstractUnitRange}})
  return Array{elt,length(axes)}
end

function default_blocks(blocktype::Type, axes::Tuple{Vararg{AbstractUnitRange}})
  block_data = Dictionary{Block{length(axes),Int},blocktype}()
  return default_blocks(block_data, axes)
end

function default_blocks(
  block_data::Dictionary{<:CartesianIndex{N},<:AbstractArray{<:Any,N}},
  axes::Tuple{Vararg{AbstractUnitRange,N}},
) where {N}
  return SparseArrayDOK(block_data, blocklength.(axes), BlockZero(axes))
end
