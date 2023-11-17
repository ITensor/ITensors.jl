function LinearAlgebra.qr(a::AbstractArray, labels_a, labels_q, labels_r)
  return qr(a, bipartitioned_permutations(qr, labels_a, labels_q, labels_r)...)
end

function LinearAlgebra.qr(a::AbstractArray, biperm::BipartitionedPermutation)
  # TODO: Use a thin QR, define `qr_thin`.
  a_matricized = matricize(a, biperm)
  q_matricized, r_matricized = qr(a_matricized)
  q_matricized_thin = typeof(a_matricized)(q_matricized)
  axes_codomain, axes_domain = bipartition(axes(a), biperm)
  q = unmatricize(q_matricized_thin, axes_codomain, (axes(q_matricized_thin, 2),))
  r = unmatricize(r_matricized, (axes(r_matricized, 1),), axes_domain)
  return q, r
end

function TensorAlgebra.bipartitioned_permutations(qr, labels_a, labels_q, labels_r)
  # TODO: Use something like `findall`?
  pos_q = map(l -> findfirst(isequal(l), labels_a), labels_q)
  pos_r = map(l -> findfirst(isequal(l), labels_a), labels_r)
  return (BipartitionedPermutation(pos_q, pos_r),)
end
