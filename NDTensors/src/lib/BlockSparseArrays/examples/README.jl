# # BlockSparseArrays.jl
# 
# A Julia `BlockSparseArray` type based on the `BlockArrays.jl` interface.
# 
# It wraps an elementwise `SparseArray` type that uses a dictionary-of-keys
# to store non-zero values, specifically a `Dictionary` from `Dictionaries.jl`.
# `BlockArrays` reinterprets the `SparseArray` as a blocked data structure.

using NDTensors.BlockSparseArrays: BlockSparseArray
using BlockArrays: BlockArrays, blockedrange
using Test: @test

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

# # BlockSparseArrays.jl and BlockArrays.jl interface

using NDTensors.BlockSparseArrays: BlockSparseArray
using BlockArrays: BlockArrays

i1 = [2, 3]
i2 = [2, 3]
B = BlockSparseArray{Float64}(i1, i2)
B[BlockArrays.Block(1, 1)] = randn(2, 2)
B[BlockArrays.Block(2, 2)] = randn(3, 3)

## Minimal interface

## Specifies the block structure
@show collect.(BlockArrays.blockaxes(axes(B, 1)))

## Index range of a block
@show axes(B, 1)[BlockArrays.Block(1)]

## Last index of each block
@show BlockArrays.blocklasts(axes(B, 1))

## Find the block containing the index
@show BlockArrays.findblock(axes(B, 1), 3)

## Retrieve a block
@show B[BlockArrays.Block(1, 1)]
@show BlockArrays.viewblock(B, BlockArrays.Block(1, 1))

## Check block bounds
@show BlockArrays.blockcheckbounds(B, 2, 2)
@show BlockArrays.blockcheckbounds(B, BlockArrays.Block(2, 2))

## Derived interface

## Specifies the block structure
@show collect(Iterators.product(BlockArrays.blockaxes(B)...))

## Iterate over block views
@show sum.(BlockArrays.eachblock(B))

## Reshape into 1-d
@show BlockArrays.blockvec(B)[BlockArrays.Block(1)]

## Array-of-array view
@show BlockArrays.blocks(B)[1, 1] == B[BlockArrays.Block(1, 1)]

## Access an index within a block
@show B[BlockArrays.Block(1, 1)[1, 1]] == B[1, 1]

# # Proposals for interfaces based on `BlockArrays.jl`, `SparseArrays`, and `BlockSparseArrays.jl`

#=
```julia
# BlockSparseArray interface

# Define `eachblockindex`
eachblockindex(B::BlockArrays.AbstractBlockArray) = Iterators.product(BlockArrays.blockaxes(B)...)

eachblockindex(B::BlockArrays.AbstractBlockArray, b::Block) # indices in a block

blocksize(B::BlockArrays.AbstractBlockArray, b::Block) # size of a block
blocksize(axes, b::Block) # size of a block

blocklength(B::BlockArrays.AbstractBlockArray, b::Block) # length of a block
blocklength(axes, b::Block) # length of a block

# Other functions
BlockArrays.blocksize(B) # number of blocks in each dimension
BlockArrays.blocksizes(B) # length of blocks in each dimension

tuple_block(Block(2, 2)) == (Block(2), Block(2)) # Block.(b.n)
blocksize(axes, b::Block) = map(axis -> length(axis[Block(b.n)]), axes)
blocksize(B, Block(2, 2)) = size(B[Block(2, 2)]) # size of a specified block

# SparseArrays interface

findnz(S) # outputs nonzero keys and values (SparseArrayKit.nonzero_pairs)
nonzeros(S) # vector of structural nonzeros (SparseArrayKit.nonzero_values)
nnz(S) # number of nonzero values (SparseArrayKit.nonzero_length)
rowvals(S) # row that each nonzero value in `nonzeros(S)` is in
nzrange(S, c) # range of linear indices into `nonzeros(S)` for values in column `c`
findall(!iszero, S) # CartesianIndices of numerical nonzeros
issparse(S)
sparse(A) # convert to sparse
dropzeros!(S)
droptol!(S, tol)

# BlockSparseArrays.jl + SparseArrays

blockfindnz(B) # outputs nonzero block indices/keys and block views
blocknonzeros(B)
blocknnz(S)
blockfindall(!iszero, B)
isblocksparse(B)
blocksparse(A)
blockdropzeros!(B)
blockdroptol!(B, tol)

# SparseArrayKit.jl interface

nonzero_pairs(a) # SparseArrays.findnz
nonzero_keys(a) # SparseArrays.?
nonzero_values(a) # SparseArrays.nonzeros
nonzero_length(a) # SparseArrays.nnz

# BlockSparseArrays.jl + SparseArrayKit.jl interface

block_nonzero_pairs
block_nonzero_keys
block_nonzero_values
block_nonzero_length
```
=#

#=
You can generate this README with:
```julia
using Literate
using NDTensors.BlockSparseArrays
dir = joinpath(pkgdir(BlockSparseArrays), "src", "BlockSparseArrays")
Literate.markdown(joinpath(dir, "examples", "README.jl"), dir; flavor=Literate.CommonMarkFlavor())
```
=#
