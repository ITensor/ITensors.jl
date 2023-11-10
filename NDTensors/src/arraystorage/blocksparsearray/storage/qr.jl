## A = sparse(I,J,V)
## rows = rowvals(A)
## vals = nonzeros(A)
## m, n = size(A)
## for j = 1:n
##    for i in nzrange(A, j)
##       row = rows[i]
##       val = vals[i]
##       # perform sparse wizardry...
##    end
## end

using SparseArrays: sparse

function sparse_structure(a::SparseArray)
  keys = Tuple.(nonzero_keys(a))
  I = first.(keys)
  J = first.(keys)
  return sparse(I, J, trues(length(keys)))
end

function LinearAlgebra.qr(a::BlockSparseArray; kwargs...)
  b = blocks(a)
  @show b
  @show nonzero_keys(b)
  structure = sparse_structure(b)
  @show structure
  return error("Not implemented")
end
