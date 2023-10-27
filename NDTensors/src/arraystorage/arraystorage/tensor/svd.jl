function LinearAlgebra.svd(tens::MatrixStorageTensor)
  F = svd(storage(tens))
  U, S, V = F.U, F.S, F.Vt
  i, j = inds(tens)
  # TODO: Make this more general with a `similar_ind` function,
  # so the dimension can be determined from the length of `S`.
  min_ij = dim(i) ≤ dim(j) ? i : j
  α = sim(min_ij) # similar_ind(i, space(S))
  β = sim(min_ij) # similar_ind(i, space(S))
  Utensor = tensor(U, (i, α))
  # TODO: Remove conversion to `Diagonal` to make more general, or make a generic `Diagonal` concept that works for `BlockSparseArray`.
  # Used for now to avoid introducing wrapper types.
  Stensor = tensor(Diagonal(S), (α, β))
  Vtensor = tensor(V, (β, j))
  return Utensor, Stensor, Vtensor, Spectrum(nothing, 0.0)
end
