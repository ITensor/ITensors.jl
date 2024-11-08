module BlockSparseArraysAdaptExt
using Adapt: Adapt, adapt
using ..BlockSparseArrays: AbstractBlockSparseArray, map_stored_blocks
Adapt.adapt_structure(to, x::AbstractBlockSparseArray) = map_stored_blocks(adapt(to), x)
end
