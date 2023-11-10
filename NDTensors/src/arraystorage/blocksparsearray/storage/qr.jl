## using SparseArrays: SparseMatrixCSC, nzrange, sparse
using BlockArrays: BlockArrays
using BlockArrays: blocklengths # Temporary, delete
using .BlockSparseArrays: nonzero_blockkeys

## # Check if the matrix has 1 or fewer entries
## # per row/column.
## function is_permutation_matrix(a::SparseMatrixCSC)
##   return all(col -> length(nzrange(a, col)) ≤ 1, axes(a, 2))
## end

## # Get the sparse structure of a SparseArray as a SparseMatrixCSC.
## function sparse_structure(a::SparseArray{<:Any,2})
##   keys = Tuple.(collect(nonzero_keys(a)))
##   I = first.(keys)
##   J = last.(keys)
##   return sparse(I, J, trues(length(keys)))
## end

# Check if the matrix has 1 or fewer entries
# per row/column.
function is_permutation_matrix(a::SparseArray{<:Any,2})
  keys = collect(Iterators.map(Tuple, nonzero_keys(a)))
  I = first.(keys)
  J = last.(keys)
  return allunique(I) && allunique(J)
end

## function block_sparse_structure(a::BlockSparseArray{<:Any,2})
##   return sparse_structure(blocks(a))
## end

function is_block_permutation_matrix(a::BlockSparseArray{<:Any,2})
  return is_permutation_matrix(blocks(a))
end

# m×n → (m×n)⋅(n×n)
q_axes(::Algorithm"thin", a::BlockSparseArray{<:Any,2}) = axes(a)
r_axes(::Algorithm"thin", a::BlockSparseArray{<:Any,2}) = (axes(a, 2), axes(a, 2))

# TODO: Is this correct? Maybe the blocks get permuted
# based on the QNs?
q_block(a::BlockSparseArray{<:Any,2}, b::BlockArrays.Block) = b
function r_block(a::BlockSparseArray{<:Any,2}, b::BlockArrays.Block)
  # TODO: Use `Int.(Tuple(b))`
  _, j = b.n
  return BlockArrays.Block(j, j)
end

# m×n → (m×m)⋅(m×n)
q_axes(::Algorithm"full", a::BlockSparseArray{<:Any,2}) = (axes(a, 1), axes(a, 1))
r_axes(::Algorithm"full", a::BlockSparseArray{<:Any,2}) = axes(a)

function LinearAlgebra.qr(a::BlockSparseArray{<:Any,2}; alg="thin")
  return qr(Algorithm(alg), a)
end

function LinearAlgebra.qr(alg::Algorithm, a::BlockSparseArray{<:Any,2})
  # Must have 1 or fewer blocks per row/column.
  if !is_block_permutation_matrix(a)
    println("Block sparsity structure is:")
    display(nonzero_blockkeys(a))
    error("Not a block permutation matrix")
  end
  eltype_a = eltype(a)
  # TODO: These axes aren't quite correct!
  axes_q = q_axes(alg, a)
  axes_r = r_axes(alg, a)
  q = BlockSparseArray{eltype_a}(axes_q)
  r = BlockSparseArray{eltype_a}(axes_r)
  for block_a in nonzero_blockkeys(a)
    # TODO: Make thin or full depending on `alg`.
    q_b, r_b = qr(a[block_a])
    # Determine the block of Q and R
    # TODO: Do the block locations change for `alg="full"`?
    block_q = q_block(a, block_a)
    block_r = r_block(a, block_a)
    q[block_q] = Matrix(q_b)
    r[block_r] = r_b
  end
  # TODO: If `alg="full"`, fill in blocks of `q`
  # with random unitaries.
  # Which blocks should be filled? Seems to be based
  # on the QNs...
  # Maybe fill diagonal blocks.
  return q, r
end
