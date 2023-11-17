function contract!(
  alg::Algorithm"matricize",
  a_dest::AbstractArray,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α,
  β,
)
  biperm_dest, biperm1, biperm2 = bipartitioned_permutations(
    contract, labels_dest, labels1, labels2
  )
  a_dest_matricized = matricize(a_dest, biperm_dest)
  a1_matricized = matricize(a1, biperm1)
  a2_matricized = matricize(a2, biperm2)
  mul!(a_dest_matricized, a1_matricized, a2_matricized, α, β)
  perm_dest = flatten(biperm_dest)
  # TODO: Create a function `unmatricize` or `unfusedims`.
  a_dest_copy = reshape(a_dest_matricized, axes(a_dest))
  permutedims!(a_dest, a_dest_copy, perm_dest)
  return a_dest
end
