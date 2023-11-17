function LinearAlgebra.qr(a::AbstractArray, labels_a, labels_q, labels_r)
  return qr(a, bipartitioned_permutations(qr, labels_a, labels_q, labels_r)...)
end

function LinearAlgebra.qr(a::AbstractArray, biperm::BipartitionedPermutation)
  @show biperm

  # TODO: Use a thin QR.
  q_matricized, r_matricized = qr(matricize(a, biperm))
  axes_codomain, axes_domain = bipartition_axes(axes(a), biperm)

  println("Q")
  @show q_matricized
  @show axes_codomain
  @show (axes(q_matricized, 2),)
  println("R")
  @show r_matricized
  @show (axes(r_matricized, 1),)
  @show axes_domain

  q = unmatricize(q_matricized, axes_codomain, (axes(q_matricized, 2),))
  r = unmatricize(r_matricized, (axes(r_matricized, 1),), axes_domain)
  return q, r
end

function TensorAlgebra.bipartitioned_permutations(qr, labels_a, labels_q, labels_r)
  # TODO: Use something like `findall`?
  pos_q = map(l -> findfirst(isequal(l), labels_a), labels_q)
  pos_r = map(l -> findfirst(isequal(l), labels_a), labels_r)
  return (BipartitionedPermutation(pos_q, pos_r),)
end

# TODO: Move to `output_labels.jl` or some other general location.
function bipartition_axes(
  axes,
  biperm::BipartitionedPermutation,
)
  axes_codomain = map(i -> axes[i], biperm[1])
  axes_domain = map(i -> axes[i], biperm[2])
  return axes_codomain, axes_domain
end
