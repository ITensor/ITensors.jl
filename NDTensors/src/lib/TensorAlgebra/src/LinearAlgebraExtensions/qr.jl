using LinearAlgebra: LinearAlgebra, qr
using ..TensorAlgebra:
  TensorAlgebra, BlockedPermutation, blockedperm, blockpermute, fusedims, splitdims

function _qr(a::AbstractArray, biperm::BlockedPermutation{2})
  a_matricized = fusedims(a, biperm)

  # TODO: Make this more generic, allow choosing thin or full,
  # make sure this works on GPU.
  q_matricized, r_matricized = qr(a_matricized)
  q_matricized_thin = typeof(a_matricized)(q_matricized)

  axes_codomain, axes_domain = blockpermute(axes(a), biperm)
  axes_q = (axes_codomain..., axes(q_matricized_thin, 2))
  biperm_q = blockedperm(
    (1:length(axes_codomain), (length(axes_codomain) + 1,)), length(axes_codomain) + 1
  )
  axes_r = (axes(r_matricized, 1), axes_domain...)
  biperm_r = blockedperm(((1,), 2:(length(axes_domain) + 1)), length(axes_domain) + 1)
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

function LinearAlgebra.qr(
  a::AbstractArray, labels_a::Tuple, labels_q::Tuple, labels_r::Tuple
)
  return qr(a, blockedperm(qr, labels_a, labels_q, labels_r))
end

# Fix ambiguity error with `LinearAlgebra`.
function LinearAlgebra.qr(
  a::AbstractMatrix, labels_a::Tuple, labels_q::Tuple, labels_r::Tuple
)
  return qr(a, blockedperm(qr, labels_a, labels_q, labels_r))
end

function TensorAlgebra.blockedperm(
  ::typeof(qr), labels_a::Tuple, labels_q::Tuple, labels_r::Tuple
)
  # TODO: Use `indexin`?
  pos_q = map(l -> findfirst(isequal(l), labels_a), labels_q)
  pos_r = map(l -> findfirst(isequal(l), labels_a), labels_r)
  return blockedperm((pos_q, pos_r), length(labels_a))
end
