# BlockSparseArrays.jl

A Julia `BlockSparseArray` type based on the `BlockArrays.jl` interface.

It wraps an elementwise `SparseArray` type that uses a dictionary-of-keys
to store non-zero values, specifically a `Dictionary` from `Dictionaries.jl`.
`BlockArrays` reinterprets the `SparseArray` as a blocked data structure.

```julia
using NDTensors.BlockSparseArrays
using BlockArrays
using Dictionaries

# Block dimensions
i1 = [2, 3]
i2 = [2, 3]

i_axes = (blockedrange(i1), blockedrange(i2))

function block_size(axes, block)
  return length.(getindex.(axes, Block.(block.n)))
end

# Data
nz_blocks = [Block(1, 1), Block(2, 2)]
nz_block_sizes = [block_size(i_axes, nz_block) for nz_block in nz_blocks]
nz_block_lengths = prod.(nz_block_sizes)

# Blocks with discontiguous underlying data
d_blocks = randn.(nz_block_sizes)

# Blocks with contiguous underlying data
# d_data = PseudoBlockVector(randn(sum(nz_block_lengths)), nz_block_lengths)
# d_blocks = [reshape(@view(d_data[Block(i)]), block_size(i_axes, nz_blocks[i])) for i in 1:length(nz_blocks)]

block_data = Dictionary([CartesianIndex(nz_block.n) for nz_block in nz_blocks], d_blocks)
block_storage = SparseArray{valtype(block_data),length(i_axes)}(block_data, blocklength.(i_axes))

B = BlockSparseArray(block_storage, i_axes)
B * B
```

## TODO

- Define an `AbstractBlockSparseArray` type along with two concrete types, one with blocks that makes no assumptions about data layout (they could be slices into contiguous data or not), and one that uses a contiguous memory in the background (which could be any `AbstractVector` wrapped in a `PseudoBlockVector` that tracks the blocks as shown above).
- Define fast linear algebra (matmul, SVD, QR, etc.) that takes advantage of sparsity.
- Define tensor contraction and addition using the `TensorOperations.jl` tensor operations interface (`tensoradd!`, `tensorcontract!`, and `tensortrace!`). See `SparseArrayKit.jl` for examples of overloading for sparse data structures.
- Use `SparseArrayKit.jl` as the elementwise sparse array backend (it would need to be generalized a little,
for example it makes the assumption that `zero` is defined for the element type, which isn't the case when the values are matrices since it would need shape information, though it could output a universal zero tensor).
- Implement `SparseArrays` functionality such as `findnz`, `findall(!iszero, B)`, `nnz`, `nonzeros`, `dropzeros`, and `droptol!`, along with the block versions of those (which would get forwarded to the `SparseArray` data structure, where they are treated as elementwise sparsity). `SparseArrayKit.jl` has functions `nonzero_pairs`, `nonzero_keys`, `nonzero_values`, and `nonzero_length` which could have analagous block functions.
- Look at other packages that deal with block sparsity such as `BlockSparseMatrices.jl` and `BlockBandedMatrices.jl` for ideas on code design and interfaces.
