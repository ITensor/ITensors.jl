using ...SparseArraysBase: SparseArrayDOK

# Check if the matrix has 1 or fewer entries
# per row/column.
function is_permutation_matrix(a::SparseMatrixCSC)
  return all(col -> length(nzrange(a, col)) ≤ 1, axes(a, 2))
end

# Check if the matrix has 1 or fewer entries
# per row/column.
function is_permutation_matrix(a::SparseArrayDOK{<:Any,2})
  keys = collect(Iterators.map(Tuple, nonzero_keys(a)))
  I = first.(keys)
  J = last.(keys)
  return allunique(I) && allunique(J)
end

function findnonzerorows(a::SparseMatrixCSC, col)
  return view(a.rowval, a.colptr[col]:(a.colptr[col + 1] - 1))
end

# TODO: Is this already defined?
function SparseArrays.SparseMatrixCSC(a::SparseArrayDOK{<:Any,2})
  # Not defined:
  # a_csc = SparseMatrixCSC{eltype(a)}(size(a))
  a_csc = spzeros(eltype(a), size(a))
  for I in nonzero_keys(a)
    a_csc[I] = a[I]
  end
  return a_csc
end

# TODO: Is this already defined?
# Get the sparse structure of a SparseArray as a SparseMatrixCSC.
function sparse_structure(
  structure_type::Type{<:SparseMatrixCSC}, a::SparseArrayDOK{<:Any,2}
)
  # Idealy would work but a bit too complicated for `map` right now:
  # return SparseMatrixCSC(map(x -> iszero(x) ? false : true, a))
  # TODO: Change to `spzeros(Bool, size(a))`.
  a_structure = structure_type(spzeros(Bool, size(a)...))
  for I in nonzero_keys(a)
    i, j = Tuple(I)
    a_structure[i, j] = true
  end
  return a_structure
end

# Get the sparsity structure as a `SparseMatrixCSC` with values
# of `true` where there are structural nonzero blocks and `false`
# otherwise.
function block_sparse_structure(structure_type::Type, a::BlockSparseArray{<:Any,2})
  return sparse_structure(structure_type, blocks(a))
end

function is_block_permutation_matrix(a::BlockSparseArray{<:Any,2})
  return is_permutation_matrix(blocks(a))
end

qr_rank(alg::Algorithm"thin", a::AbstractArray{<:Any,2}) = minimum(size(a))

# m × n → (m × min(m, n)) ⋅ (min(m, n) × n)
function qr_block_sparse_structure(alg::Algorithm"thin", a::BlockSparseArray{<:Any,2})
  axes_row, axes_col = axes(a)
  a_csc = block_sparse_structure(SparseMatrixCSC, a)
  F = qr(float(a_csc))
  # Outputs full Q
  # q_csc = sparse(F.Q[invperm(F.prow), :])
  q_csc = (F.Q * sparse(I, size(a_csc, 1), minimum(size(a_csc))))[invperm(F.prow), :]
  r_csc = F.R[:, invperm(F.pcol)]
  nblocks = size(q_csc, 2)
  @assert nblocks == size(r_csc, 1)
  a_sparse = blocks(a)
  blocklengths_qr = Vector{Int}(undef, nblocks)
  for I in nonzero_keys(a_sparse)
    i, k = Tuple(I)
    # Get the nonzero columns associated
    # with the given row.
    j = only(findnonzerorows(r_csc, k))
    # @assert is_structural_nonzero(r, j, k)
    # @assert is_structural_nonzero(q, i, j)
    blocklengths_qr[j] = qr_rank(alg, @view(a[BlockArrays.Block(i, k)]))
  end
  axes_qr = blockedrange(blocklengths_qr)
  axes_q = (axes(a, 1), axes_qr)
  axes_r = (axes_qr, axes(a, 2))
  # TODO: Come up with a better format to ouput.
  # TODO: Get `axes_qr` as a permutation of the
  # axes of `axes(a, 2)` to preserve sectors
  # when using symmetric tensors.
  return q_csc, axes_q, r_csc, axes_r
end

# m × n → (m × m) ⋅ (m × n)
function qr_block_sparse_structure(alg::Algorithm"full", a::BlockSparseArray{<:Any,2})
  return error("Not implemented")
end

function qr_blocks(a, structure_r, block_a)
  i, k = block_a.n
  j = only(findnonzerorows(structure_r, k))
  return BlockArrays.Block(i, j), BlockArrays.Block(j, k)
end

# Block-preserving QR.
function LinearAlgebra.qr(a::BlockSparseArray{<:Any,2}; alg="thin")
  return qr(Algorithm(alg), a)
end

# Block-preserving QR.
function LinearAlgebra.qr(alg::Algorithm, a::BlockSparseArray{<:Any,2})
  if !is_block_permutation_matrix(a)
    # Must have 1 or fewer blocks per row/column.
    println("Block sparsity structure is:")
    display(nonzero_blockkeys(a))
    error("Not a block permutation matrix")
  end
  eltype_a = eltype(a)
  # TODO: `structure_q` isn't needed.
  structure_q, axes_q, structure_r, axes_r = qr_block_sparse_structure(alg, a)
  # TODO: Make this generic to GPU, use `similar`.
  q = BlockSparseArray{eltype_a}(axes_q)
  r = BlockSparseArray{eltype_a}(axes_r)
  for block_a in nonzero_blockkeys(a)
    # TODO: Make thin or full depending on `alg`.
    q_b, r_b = qr(a[block_a])
    # Determine the block of Q and R
    # TODO: Do the block locations change for `alg="full"`?
    block_q, block_r = qr_blocks(a, structure_r, block_a)

    # TODO Make this generic to GPU.
    q[block_q] = Matrix(q_b)
    r[block_r] = r_b
  end
  # TODO: If `alg="full"`, fill in blocks of `q`
  # with random unitaries.
  # Which blocks should be filled? Seems to be based
  # on the QNs...
  # Maybe fill diagonal blocks.
  # TODO: Also store `structure_r` in some way
  # since that is needed for permuting the QNs.
  return q, r
end
