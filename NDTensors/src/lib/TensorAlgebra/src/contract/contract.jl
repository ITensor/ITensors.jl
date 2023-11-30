# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.

default_contract_alg() = Algorithm"matricize"()

function contract(a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...)
  return contract(a1, labels1, a2, labels2, true, false; kwargs...)
end

function contract(
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α,
  β;
  alg=default_contract_alg(),
  kwargs...,
)
  return contract(Algorithm(alg), a1, labels1, a2, labels2, α, β; kwargs...)
end

function contract(
  alg::Algorithm, a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...
)
  return contract(alg, a1, labels1, a2, labels2, true, false; kwargs...)
end

function contract(
  alg::Algorithm, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β; kwargs...
)
  labels_dest = output_labels(contract, alg, a1, labels1, a2, labels2, α, β; kwargs...)
  return contract(alg, labels_dest, a1, labels1, a2, labels2, α, β; kwargs...), labels_dest
end

function contract(
  labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...
)
  return contract(
    labels_dest,
    a1::AbstractArray,
    labels1,
    a2::AbstractArray,
    labels2,
    true,
    false;
    kwargs...,
  )
end

function contract(
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α,
  β;
  alg=default_contract_alg(),
  kwargs...,
)
  return contract(Algorithm(alg), labels_dest, a1, labels1, a2, labels2, α, β; kwargs...)
end

function contract(
  alg::Algorithm,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2;
  kwargs...,
)
  return contract(alg, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...)
end

function contract(
  alg::Algorithm,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α,
  β;
  kwargs...,
)
  biperm_dest, biperm1, biperm2 = bipartitioned_permutations(
    contract, alg, labels_dest, labels1, labels2
  )
  a_dest = allocate_output(
    contract, alg, biperm_dest, a1, biperm1, a2, biperm2, α, β; kwargs...
  )
  contract!(alg, a_dest, biperm_dest, a1, biperm1, a2, biperm2, α, β; kwargs...)
  return a_dest
end

function contract!(
  a_dest::AbstractArray,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2;
  kwargs...,
)
  contract!(a_dest, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...)
  return a_dest
end

function contract!(
  a_dest::AbstractArray,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α,
  β;
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
  labels2;
  kwargs...,
)
  contract!(alg, a_dest, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...)
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
