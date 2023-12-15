# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.

default_contract_alg() = Algorithm"matricize"()

function contract(
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α=true;
  alg=default_contract_alg(),
  kwargs...,
)
  return contract(Algorithm(alg), a1, labels1, a2, labels2, α; kwargs...)
end

function contract(
  alg::Algorithm, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α=true; kwargs...
)
  labels_dest = output_labels(contract, alg, a1, labels1, a2, labels2, α; kwargs...)
  return contract(alg, labels_dest, a1, labels1, a2, labels2, α; kwargs...), labels_dest
end

function contract(
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α=true;
  alg=default_contract_alg(),
  kwargs...,
)
  return contract(Algorithm(alg), labels_dest, a1, labels1, a2, labels2, α; kwargs...)
end

function contract!(
  a_dest::AbstractArray,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α=true,
  β=false;
  alg=default_contract_alg(),
  kwargs...,
)
  contract!(Algorithm(alg), a_dest, labels_dest, a1, labels1, a2, labels2, α, β; kwargs...)
  return a_dest
end

function contract!(
  alg::Algorithm,
  a_dest::AbstractArray,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α,
  β;
  kwargs...,
)
  return error("Not implemented")
end

function contract(
  alg::Algorithm,
  biperm_dest::BlockedPermutation{2},
  a1::AbstractArray,
  biperm1::BlockedPermutation{2},
  a2::AbstractArray,
  biperm2::BlockedPermutation{2},
  α;
  kwargs...,
)
  return error("Not implemented")
end
