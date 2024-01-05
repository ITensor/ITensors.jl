using ..AlgorithmSelection: Algorithm, @Algorithm_str

# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.

default_contract_alg() = Algorithm"matricize"()

# Required interface if not using
# matricized contraction.
function contract!(
  alg::Algorithm,
  a_dest::AbstractArray,
  biperm_dest::BlockedPermutation,
  a1::AbstractArray,
  biperm1::BlockedPermutation,
  a2::AbstractArray,
  biperm2::BlockedPermutation,
  α::Number,
  β::Number,
)
  return error("Not implemented")
end

function contract(
  a1::AbstractArray,
  labels1::Tuple,
  a2::AbstractArray,
  labels2::Tuple,
  α::Number=true;
  alg=default_contract_alg(),
  kwargs...,
)
  return contract(Algorithm(alg), a1, labels1, a2, labels2, α; kwargs...)
end

function contract(
  alg::Algorithm,
  a1::AbstractArray,
  labels1::Tuple,
  a2::AbstractArray,
  labels2::Tuple,
  α::Number=true;
  kwargs...,
)
  labels_dest = output_labels(contract, alg, a1, labels1, a2, labels2, α; kwargs...)
  return contract(alg, labels_dest, a1, labels1, a2, labels2, α; kwargs...), labels_dest
end

function contract(
  labels_dest::Tuple,
  a1::AbstractArray,
  labels1::Tuple,
  a2::AbstractArray,
  labels2::Tuple,
  α::Number=true;
  alg=default_contract_alg(),
  kwargs...,
)
  return contract(Algorithm(alg), labels_dest, a1, labels1, a2, labels2, α; kwargs...)
end

function contract!(
  a_dest::AbstractArray,
  labels_dest::Tuple,
  a1::AbstractArray,
  labels1::Tuple,
  a2::AbstractArray,
  labels2::Tuple,
  α::Number=true,
  β::Number=false;
  alg=default_contract_alg(),
  kwargs...,
)
  contract!(Algorithm(alg), a_dest, labels_dest, a1, labels1, a2, labels2, α, β; kwargs...)
  return a_dest
end

function contract(
  alg::Algorithm,
  labels_dest::Tuple,
  a1::AbstractArray,
  labels1::Tuple,
  a2::AbstractArray,
  labels2::Tuple,
  α::Number=true;
  kwargs...,
)
  biperm_dest, biperm1, biperm2 = blockedperms(contract, labels_dest, labels1, labels2)
  return contract(alg, biperm_dest, a1, biperm1, a2, biperm2, α; kwargs...)
end

function contract!(
  alg::Algorithm,
  a_dest::AbstractArray,
  labels_dest::Tuple,
  a1::AbstractArray,
  labels1::Tuple,
  a2::AbstractArray,
  labels2::Tuple,
  α::Number,
  β::Number;
  kwargs...,
)
  biperm_dest, biperm1, biperm2 = blockedperms(contract, labels_dest, labels1, labels2)
  return contract!(alg, a_dest, biperm_dest, a1, biperm1, a2, biperm2, α, β; kwargs...)
end

function contract(
  alg::Algorithm,
  biperm_dest::BlockedPermutation,
  a1::AbstractArray,
  biperm1::BlockedPermutation,
  a2::AbstractArray,
  biperm2::BlockedPermutation,
  α::Number;
  kwargs...,
)
  a_dest = allocate_output(contract, biperm_dest, a1, biperm1, a2, biperm2, α)
  contract!(alg, a_dest, biperm_dest, a1, biperm1, a2, biperm2, α, false; kwargs...)
  return a_dest
end
