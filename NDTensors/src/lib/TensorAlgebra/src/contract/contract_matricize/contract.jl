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

function contract(
  alg::Algorithm"matricize",
  biperm_dest::BlockedPermutation{2},
  a1::AbstractArray,
  biperm1::BlockedPermutation{2},
  a2::AbstractArray,
  biperm2::BlockedPermutation{2},
  α,
)
  a1_mat = fusedims(a1, biperm1)
  a2_mat = fusedims(a2, biperm2)
  # TODO: Handle `α` lazily.
  a_dest_mat = α * a1_mat * a2_mat

  error()
  axes_dest # = ...
  a_dest = splitdims(a_dest_mat, axes_dest, biperm_dest)
  return a_dest
end
