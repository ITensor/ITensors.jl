using BlockArrays: BlockArrays, AbstractBlockArray

# TODO: Delete this. This function was replaced
# by `nstored` but is still used in `NDTensors`.
function nonzero_keys end

abstract type AbstractBlockSparseArray{T,N} <: AbstractBlockArray{T,N} end

# Base `AbstractArray` interface
Base.axes(::AbstractBlockSparseArray) = error("Not implemented")

# BlockArrays `AbstractBlockArray` interface
BlockArrays.blocks(::AbstractBlockSparseArray) = error("Not implemented")

# `AbstractBlockSparseArray` interface
blocktype(::AbstractBlockSparseArray) = error("Not implemented")
