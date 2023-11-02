# BlockSparseArrays.jl

A Julia `BlockSparseArray` type based on the `BlockArrays.jl` interface.

It wraps an elementwise `SparseArray` type that uses a dictionary-of-keys
to store non-zero values, specifically a `Dictionary` from `Dictionaries.jl`.
`BlockArrays` reinterprets the `SparseArray` as a blocked data structure.

````julia
using NDTensors.BlockSparseArrays
using BlockArrays: BlockArrays, blockedrange
using Test

function main()
  # Block dimensions
  i1 = [2, 3]
  i2 = [2, 3]

  i_axes = (blockedrange(i1), blockedrange(i2))

  function block_size(axes, block)
    return length.(getindex.(axes, BlockArrays.Block.(block.n)))
  end

  # Data
  nz_blocks = BlockArrays.Block.([(1, 1), (2, 2)])
  nz_block_sizes = [block_size(i_axes, nz_block) for nz_block in nz_blocks]
  nz_block_lengths = prod.(nz_block_sizes)

  # Blocks with discontiguous underlying data
  d_blocks = randn.(nz_block_sizes)

  # Blocks with contiguous underlying data
  # d_data = PseudoBlockVector(randn(sum(nz_block_lengths)), nz_block_lengths)
  # d_blocks = [reshape(@view(d_data[Block(i)]), block_size(i_axes, nz_blocks[i])) for i in 1:length(nz_blocks)]

  B = BlockSparseArray(nz_blocks, d_blocks, i_axes)

  # Access a block
  B[BlockArrays.Block(1, 1)]

  # Access a non-zero block, returns a zero matrix
  B[BlockArrays.Block(1, 2)]

  # Set a zero block
  B[BlockArrays.Block(1, 2)] = randn(2, 3)

  # Matrix multiplication (not optimized for sparsity yet)
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
````

You can generate this README with:
```julia
using Literate
using NDTensors.BlockSparseArrays
dir = joinpath(pkgdir(BlockSparseArrays), "src", "BlockSparseArrays")
Literate.markdown(joinpath(dir, "examples", "README.jl"), dir; flavor=Literate.CommonMarkFlavor())
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

