using Base.PermutedDimsArrays: genperm

# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
function output_axes(
  ::typeof(contract),
  biperm_dest::BlockedPermutation{2},
  a1::AbstractArray,
  biperm1::BlockedPermutation{2},
  a2::AbstractArray,
  biperm2::BlockedPermutation{2},
  α::Number=true,
)
  axes_codomain, axes_contracted = blockpermute(axes(a1), biperm1)
  axes_contracted2, axes_domain = blockpermute(axes(a2), biperm2)
  @assert axes_contracted == axes_contracted2
  return genperm((axes_codomain..., axes_domain...), invperm(Tuple(biperm_dest)))
end

# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
function allocate_output(
  ::typeof(contract),
  biperm_dest::BlockedPermutation{2},
  a1::AbstractArray,
  biperm1::BlockedPermutation{2},
  a2::AbstractArray,
  biperm2::BlockedPermutation{2},
  α::Number=true,
)
  axes_dest = output_axes(contract, biperm_dest, a1, biperm1, a2, biperm2, α)
  return similar(a1, promote_type(eltype(a1), eltype(a2), typeof(α)), axes_dest)
end
