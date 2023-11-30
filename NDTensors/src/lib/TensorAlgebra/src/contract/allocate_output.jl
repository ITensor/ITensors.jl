function allocate_output(
  ::typeof(contract),
  alg::Algorithm,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α,
  β,
)
  axes_dest = output_axes(contract, alg, labels_dest, axes(a1), labels1, axes(a2), labels2)
  # TODO: Define `output_type(contract, alg, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β)`.
  # TODO: Define `output_structure(contract, alg, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β)`.
  # TODO: Define `allocate(type, structure)`.
  return Array{promote_type(eltype(a1), eltype(a2))}(undef, length.(axes_dest))
end

# TODO: Generalize to `output_structure`.
function output_axes(
  f::typeof(contract), alg::Algorithm, labels_dest, axes1, labels1, axes2, labels2
)
  biperm_dest, biperm1, biperm2 = bipartitioned_permutations(
    f, labels_dest, labels1, labels2
  )
  return output_axes(f, alg, biperm_dest, axes1, biperm1, axes2, biperm2)
end

# TODO: Generalize to `output_structure`.
function output_axes(
  f::typeof(contract),
  alg::Algorithm,
  biperm_dest::BipartitionedPermutation,
  axes1,
  biperm1::BipartitionedPermutation,
  axes2,
  biperm2::BipartitionedPermutation,
)
  perm_dest = flatten(biperm_dest)
  nuncontracted1 = length(biperm1[1])
  axes_dest = map(perm_dest) do i
    return if i <= nuncontracted1
      axes1[biperm1[1][i]]
    else
      axes2[biperm2[2][i - nuncontracted1]]
    end
  end
  return axes_dest
end
