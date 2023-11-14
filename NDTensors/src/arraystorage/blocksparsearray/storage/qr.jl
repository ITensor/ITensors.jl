using SparseArrays: SparseArrays, SparseMatrixCSC, spzeros, sparse
using BlockArrays: BlockArrays, blockedrange
using BlockArrays: blocklengths # Temporary, delete
using .BlockSparseArrays: nonzero_blockkeys

## # Check if the matrix has 1 or fewer entries
## # per row/column.
## function is_permutation_matrix(a::SparseMatrixCSC)
##   return all(col -> length(nzrange(a, col)) ≤ 1, axes(a, 2))
## end

## row_to_col: [0, 3, 1, 2, 0]
## col_to_row: invperm(nonzeros(row_to_col))
## col_to_row = zero(row_to_col)
## nzrows = findall(!iszero, row_to_col)
## col_to_row[nzrows] = invperm(@view(row_to_col[nzrows]))
##
## struct PermutationMatrix <: AbstractSparseMatrix{Bool,Int}
##   row_to_col::Vector{Int}
##   col_to_row::Vector{Int}
## end

## function Graphs.outneighbors(g::SimpleWeightedGraph, v::Integer)
##   mat = g.weights
##   return view(mat.rowval, mat.colptr[v]:(mat.colptr[v + 1] - 1))
## end

function nzcols(a::SparseMatrixCSC, row)
  return view(a.rowval, a.colptr[row]:(a.colptr[row + 1] - 1))
end

function nzcolvals(a::SparseMatrixCSC, row)
  return view(nonzeros(a), nzrange(a, row))
end

function SparseArrays.SparseMatrixCSC(a::SparseArray{<:Any,2})
  # Not defined:
  # a_csc = SparseMatrixCSC{eltype(a)}(size(a))
  a_csc = spzeros(eltype(a), size(a))
  for I in nonzero_keys(a)
    a_csc[I] = a[I]
  end
  return a_csc
end

# Get the sparse structure of a SparseArray as a SparseMatrixCSC.
function sparse_structure(a::SparseArray{<:Any,2})
  return SparseMatrixCSC(map(x -> iszero(x) ? false : true, a))
end

# Check if the matrix has 1 or fewer entries
# per row/column.
# TODO: Define for `SparseMatrixCSC`, use `eachnz`?
function is_permutation_matrix(a::SparseArray{<:Any,2})
  keys = collect(Iterators.map(Tuple, nonzero_keys(a)))
  I = first.(keys)
  J = last.(keys)
  return allunique(I) && allunique(J)
end

# Get the sparsity structure as a `SparseMatrixCSC` with values
# of `true` where there are structural nonzero blocks and `false`
# otherwise.
function block_sparse_structure(a::BlockSparseArray{<:Any,2})
  return sparse_structure(blocks(a))
end

function is_block_permutation_matrix(a::BlockSparseArray{<:Any,2})
  return is_permutation_matrix(blocks(a))
end

# m × n → (m × min(m, n)) ⋅ (min(m, n) × n)
function qr_block_sparse_structure(alg::Algorithm"thin", a::BlockSparseArray{<:Any,2})
  axes_row, axes_col = axes(a)
  a_csc = block_sparse_structure(a)
  F = qr(float(a_csc))

  @show typeof(F.Q)
  @show typeof(SparseArrays.sparse(F.Q))

  q = sparse(F.Q[invperm(F.prow), :])
  r = F.R[:, invperm(F.pcol)]

  @show a_csc ≈ q * r

  display(q)
  display(r)

  nblocks = size(q, 2)
  error("Not implemented")
end

## # TODO: Is this correct? Maybe the blocks get permuted
## # based on the QNs?
## q_block(a::BlockSparseArray{<:Any,2}, b::BlockArrays.Block) = b
## function r_block(a::BlockSparseArray{<:Any,2}, b::BlockArrays.Block)
##   # TODO: Use `Int.(Tuple(b))`
##   _, j = b.n
##   return BlockArrays.Block(j, j)
## end
## 
## # m×n → (m×m)⋅(m×n)
## q_axes(::Algorithm"full", a::BlockSparseArray{<:Any,2}) = error("Not implemented") # (axes(a, 1), axes(a, 1))
## r_axes(::Algorithm"full", a::BlockSparseArray{<:Any,2}) = error("Not implemented") # axes(a)

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

  structure_q, structure_r = qr_block_sparse_structure(alg, a)
  # TODO: These axes aren't quite correct!
  axes_q = (axes(a, 1), axis_qr)
  axes_r = (axis_qr, axes(a, 2))

  @show blocklengths.(axes_q)
  @show blocklengths.(axes_r)

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
