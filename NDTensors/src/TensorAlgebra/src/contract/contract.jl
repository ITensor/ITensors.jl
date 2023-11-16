default_contract_alg() = "matricize"

function contract(a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...)
  return contract(a1, labels1, a2, labels2, true, false; kwargs...)
end

function contract(a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β; alg=default_contract_alg(), kwargs...)
  return contract(Algorithm(alg), a1, labels1, a2, labels2, α, β; kwargs...)
end

function contract(alg::Algorithm, a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...)
  return contract(alg, a1, labels1, a2, labels2, true, false; kwargs...)
end

function contract(alg::Algorithm, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β; kwargs...)
  labels_dest = contract_output_labels(alg, a1, labels1, a2, labels2, α, β; kwargs...)
  return contract(alg, labels_dest, a1, labels1, a2, labels2, α, β; kwargs...), labels_dest
end

function contract(labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...)
  return contract(labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2, true, false; kwargs...)
end

function contract(labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β; alg=default_contract_alg(), kwargs...)
  return contract(Algorithm(alg), labels_dest, a1, labels1, a2, labels2, α, β; kwargs...)
end

function contract(alg::Algorithm, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...)
  return contract(alg, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...)
end

function contract(alg::Algorithm, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β; kwargs...)
  a_dest = allocate_output(contract, alg, labels_dest, a1, labels1, a2, labels2, α, β; kwargs...)
  contract!(alg, a_dest, labels_dest, a1, labels1, a2, labels2, α, β; kwargs...)
  return a_dest
end

function contract!(a_dest::AbstractArray, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...)
  contract!(a_dest, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...)
  return a_dest
end

function contract!(a_dest::AbstractArray, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β; alg=default_contract_alg(), kwargs...)
  contract!(Algorithm(alg), a_dest, labels_dest, a1, labels1, a2, labels2, α, β; kwargs...)
  return a_dest
end

function contract!(alg::Algorithm, a_dest::AbstractArray, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...)
  contract!(alg, a_dest, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...)
  return a_dest
end

function contract!(alg::Algorithm, a_dest::AbstractArray, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β; kwargs...)
  return error("Not implemented")
end

# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.

# Helper functions
function contract_output_labels(alg::Algorithm, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β)
  return contract_output_labels(alg, labels1, labels2)
end

function contract_output_labels(alg::Algorithm, labels1, labels2)
  return contract_output_labels(labels1, labels2)
end

function contract_output_labels(labels1, labels2)
  return symdiff(labels1, labels2)
end

function allocate_output(::typeof(contract), alg::Algorithm, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β; kwargs...)
  axes1 = axes(a1)
  axes2 = axes(a2)
  x = map(label_dest -> findfirst(==(label_dest), labels1), labels_dest)
  y = map(label_dest -> findfirst(==(label_dest), labels2), labels_dest)
  @show x
  @show y
  return error("Not implemented")
end
