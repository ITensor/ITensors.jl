fuse(a1::AbstractUnitRange, a2::AbstractUnitRange) = Base.OneTo(length(a1) * length(a2))
fuse(a...) = foldl(fuse, a)

matricize(a::AbstractArray, biperm) = matricize(a, BipartitionedPermutation(biperm...))

function matricize(a::AbstractArray, biperm::BipartitionedPermutation)
  # Permute and fuse the axes
  axes_src = axes(a)
  axes_codomain = map(i -> axes_src[i], biperm[1])
  axes_domain = map(i -> axes_src[i], biperm[2])
  axis_codomain_fused = fuse(axes_codomain...)
  axis_domain_fused = fuse(axes_domain...)
  # Permute the array
  perm = flatten(biperm)
  a_permuted = permutedims(a, perm)
  return reshape(a_permuted, (axis_codomain_fused, axis_domain_fused))
end
