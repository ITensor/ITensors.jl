function contract!(
  alg::Algorithm"matricize",
  a_dest::AbstractArray,
  biperm_dest::BlockedPermutation{2},
  a1::AbstractArray,
  biperm1::BlockedPermutation{2},
  a2::AbstractArray,
  biperm2::BlockedPermutation{2},
  α,
  β,
)
  a_dest_mat = fusedims(a_dest, biperm_dest)
  a1_mat = fusedims(a1, biperm1)
  a2_mat = fusedims(a2, biperm2)
  mul!(a_dest_mat, a1_mat, a2_mat, α, β)
  splitdims!(a_dest, a_dest_mat, biperm_dest)
  return a_dest
end

# TODO: Rewrite this in terms of `contract!`.
function contract(
  alg::Algorithm"matricize",
  biperm_dest::BlockedPermutation{2},
  a1::AbstractArray,
  biperm1::BlockedPermutation{2},
  a2::AbstractArray,
  biperm2::BlockedPermutation{2},
  α,
)
  # TODO: Wrap in `allocated_output(contract, ...)`.
  axes_codomain, axes_contracted = blockpermute(axes(a1), biperm1)
  axes_contracted2, axes_domain = blockpermute(axes(a2), biperm2)
  axes_dest = (axes_codomain..., axes_domain...)
  a_dest = similar(a1, promote_type(eltype(a1), eltype(a2)), axes_dest)
  contract!(alg, a_dest, biperm_dest, a1, biperm1, a2, biperm2, α, false)
  return a_dest
end
