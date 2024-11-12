using ArrayLayouts: LayoutMatrix
using LinearAlgebra: LinearAlgebra, qr
using ..TensorAlgebra:
  TensorAlgebra,
  BlockedPermutation,
  blockedperm,
  blockedperm_indexin,
  blockpermute,
  fusedims,
  splitdims

# TODO: Define as `tensor_qr`.
# TODO: This look generic but doesn't work for `BlockSparseArrays`.
function _qr(a::AbstractArray, biperm::BlockedPermutation{2})
  a_matricized = fusedims(a, biperm)

  # TODO: Make this more generic, allow choosing thin or full,
  # make sure this works on GPU.
  q_matricized, r_matricized = qr(a_matricized)
  q_matricized_thin = typeof(a_matricized)(q_matricized)

  axes_codomain, axes_domain = blockpermute(axes(a), biperm)
  axes_q = (axes_codomain..., axes(q_matricized_thin, 2))
  # TODO: Use `tuple_oneto(n) = ntuple(identity, n)`, currently in `BlockSparseArrays`.
  biperm_q = blockedperm(
    ntuple(identity, length(axes_codomain)), (length(axes_codomain) + 1,)
  )
  axes_r = (axes(r_matricized, 1), axes_domain...)
  biperm_r = blockedperm((1,), ntuple(identity, length(axes_domain)) .+ 1)
  q = splitdims(q_matricized_thin, axes_q)
  r = splitdims(r_matricized, axes_r)
  return q, r
end

function LinearAlgebra.qr(a::AbstractArray, biperm::BlockedPermutation{2})
  return _qr(a, biperm)
end

# Fix ambiguity error with `LinearAlgebra`.
function LinearAlgebra.qr(a::AbstractMatrix, biperm::BlockedPermutation{2})
  return _qr(a, biperm)
end

# Fix ambiguity error with `ArrayLayouts`.
function LinearAlgebra.qr(a::LayoutMatrix, biperm::BlockedPermutation{2})
  return _qr(a, biperm)
end

# TODO: Define in terms of an inner function `_qr` or `tensor_qr`.
function LinearAlgebra.qr(
  a::AbstractArray, labels_a::Tuple, labels_q::Tuple, labels_r::Tuple
)
  return qr(a, blockedperm_indexin(labels_a, labels_q, labels_r))
end

# Fix ambiguity error with `LinearAlgebra`.
function LinearAlgebra.qr(
  a::AbstractMatrix, labels_a::Tuple, labels_q::Tuple, labels_r::Tuple
)
  return qr(a, blockedperm_indexin(labels_a, labels_q, labels_r))
end

# Fix ambiguity error with `ArrayLayouts`.
function LinearAlgebra.qr(
  a::LayoutMatrix, labels_a::Tuple, labels_q::Tuple, labels_r::Tuple
)
  return qr(a, blockedperm_indexin(labels_a, labels_q, labels_r))
end
