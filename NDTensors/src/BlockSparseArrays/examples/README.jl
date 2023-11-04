# # BlockSparseArrays.jl
# 
# A Julia `BlockSparseArray` type based on the `BlockArrays.jl` interface.
# 
# It wraps an elementwise `SparseArray` type that uses a dictionary-of-keys
# to store non-zero values, specifically a `Dictionary` from `Dictionaries.jl`.
# `BlockArrays` reinterprets the `SparseArray` as a blocked data structure.

using NDTensors.BlockSparseArrays
using BlockArrays: BlockArrays, blockedrange
using Test

function main()
  ## Block dimensions
  i1 = [2, 3]
  i2 = [2, 3]

  i_axes = (blockedrange(i1), blockedrange(i2))

  function block_size(axes, block)
    return length.(getindex.(axes, BlockArrays.Block.(block.n)))
  end

  ## Data
  nz_blocks = BlockArrays.Block.([(1, 1), (2, 2)])
  nz_block_sizes = [block_size(i_axes, nz_block) for nz_block in nz_blocks]
  nz_block_lengths = prod.(nz_block_sizes)

  ## Blocks with discontiguous underlying data
  d_blocks = randn.(nz_block_sizes)

  ## Blocks with contiguous underlying data
  ## d_data = PseudoBlockVector(randn(sum(nz_block_lengths)), nz_block_lengths)
  ## d_blocks = [reshape(@view(d_data[Block(i)]), block_size(i_axes, nz_blocks[i])) for i in 1:length(nz_blocks)]

  B = BlockSparseArray(nz_blocks, d_blocks, i_axes)

  ## Access a block
  B[BlockArrays.Block(1, 1)]

  ## Access a non-zero block, returns a zero matrix
  B[BlockArrays.Block(1, 2)]

  ## Set a zero block
  B[BlockArrays.Block(1, 2)] = randn(2, 3)

  ## Matrix multiplication (not optimized for sparsity yet)
  @test B * B ≈ Array(B) * Array(B)

  permuted_B = permutedims(B, (2, 1))
  @test permuted_B isa BlockSparseArray
  @test permuted_B == permutedims(Array(B), (2, 1))

  @test B + B ≈ Array(B) + Array(B)
  @test 2B ≈ 2Array(B)

  @test reshape(B, ([4, 6, 6, 9],)) isa BlockSparseArray{<:Any,1}

  return nothing
end

main()

# ## BlockSparseArrays.jl and BlockArrays.jl interface

using NDTensors.BlockSparseArrays
using BlockArrays: BlockArrays

i1 = [2, 3]
i2 = [2, 3]
B = BlockSparseArray{Float64}(i1, i2)
B[BlockArrays.Block(1, 1)] = randn(2, 2)
B[BlockArrays.Block(2, 2)] = randn(3, 3)

# Minimal interface

# Specifies the block structure
@show collect.(BlockArrays.blockaxes(axes(B, 1)))

# Index range of a block
@show axes(B, 1)[BlockArrays.Block(1)]

# Last index of each block
@show BlockArrays.blocklasts(axes(B, 1))

# Find the block containing the index
@show BlockArrays.findblock(axes(B, 1), 3)

# Retrieve a block
@show B[BlockArrays.Block(1, 1)]
@show BlockArrays.viewblock(B, BlockArrays.Block(1, 1))

# Check block bounds
@show BlockArrays.blockcheckbounds(B, 2, 2)
@show BlockArrays.blockcheckbounds(B, BlockArrays.Block(2, 2))

# Derived interface

# Specifies the block structure
@show collect(Iterators.product(BlockArrays.blockaxes(B)...))

# Iterate over block views
@show sum.(BlockArrays.eachblock(B))

# Reshape into 1-d
@show BlockArrays.blockvec(B)[BlockArrays.Block(1)]

# Array-of-array view
@show BlockArrays.blocks(B)[1, 1] == B[BlockArrays.Block(1, 1)]

# Access an index within a block
@show B[BlockArrays.Block(1, 1)[1, 1]] == B[1, 1]

# BlockSparseArray interface

# Define `eachblockindex`
eachblockindex(B::BlockArrays.AbstractBlockArray) = Iterators.product(BlockArrays.blockaxes(B)...)
eachblockindex(B::BlockSparseArray) = Iterators.product(BlockArrays.blockaxes(B)...)

#=
You can generate this README with:
```julia
using Literate
using NDTensors.BlockSparseArrays
dir = joinpath(pkgdir(BlockSparseArrays), "src", "BlockSparseArrays")
Literate.markdown(joinpath(dir, "examples", "README.jl"), dir; flavor=Literate.CommonMarkFlavor())
```
=#
