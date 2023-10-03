# # BlockSparseArrays.jl
# 
# A Julia `BlockSparseArray` type based on the `BlockArrays.jl` interface.
# 
# It wraps an elementwise `SparseArray` type that uses a dictionary-of-keys
# to store non-zero values, specifically a `Dictionary` from `Dictionaries.jl`.
# `BlockArrays` reinterprets the `SparseArray` as a blocked data structure.

using NDTensors.BlockSparseArrays
using BlockArrays

## Block dimensions
i1 = [2, 3]
i2 = [2, 3]

i_axes = (blockedrange(i1), blockedrange(i2))

function block_size(axes, block)
  return length.(getindex.(axes, Block.(block.n)))
end

## Data
nz_blocks = [Block(1, 1), Block(2, 2)]
nz_block_sizes = [block_size(i_axes, nz_block) for nz_block in nz_blocks]
nz_block_lengths = prod.(nz_block_sizes)

## Blocks with discontiguous underlying data
d_blocks = randn.(nz_block_sizes)

## Blocks with contiguous underlying data
## d_data = PseudoBlockVector(randn(sum(nz_block_lengths)), nz_block_lengths)
## d_blocks = [reshape(@view(d_data[Block(i)]), block_size(i_axes, nz_blocks[i])) for i in 1:length(nz_blocks)]

B = BlockSparseArray(nz_blocks, d_blocks, i_axes)

## Access a block
B[Block(1, 1)]

## Access a non-zero block, returns a zero matrix
B[Block(1, 2)]

## Set a zero block
B[Block(1, 2)] = randn(2, 3)

## Matrix multiplication (not optimized for sparsity yet)
B * B

# You can generate this README with:
# ```julia
# using Literate
# Literate.markdown("examples/README.jl", "."; flavor=Literate.CommonMarkFlavor())
# ```
