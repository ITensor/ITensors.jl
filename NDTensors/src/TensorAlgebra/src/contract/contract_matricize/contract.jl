function contract!(
  alg::Algorithm"matricize",
  a_dest::AbstractArray,
  biperm_dest::BipartitionedPermutation,
  a1::AbstractArray,
  biperm1::BipartitionedPermutation,
  a2::AbstractArray,
  biperm2::BipartitionedPermutation,
  α,
  β,
)
  a_dest_matricized = matricize(a_dest, biperm_dest)
  a1_matricized = matricize(a1, biperm1)
  a2_matricized = matricize(a2, biperm2)
  mul!(a_dest_matricized, a1_matricized, a2_matricized, α, β)
  perm_dest = flatten(biperm_dest)
  # TODO: Create a function `unmatricize` or `unfusedims`.
  # unmatricize!(a_dest, a_dest_matricized, axes(a_dest), perm_dest)
  a_dest_copy = reshape(a_dest_matricized, axes(a_dest))
  permutedims!(a_dest, a_dest_copy, perm_dest)
  return a_dest
end
