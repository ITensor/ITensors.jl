using ITensors
using Symbolics

using ITensors: contraction_cost, optimal_contraction_sequence

function tensor_network(; m, k, d)
  l = Index(m, "l")
  r = Index(m, "r")
  h₁ = Index(k, "h₁")
  h₂ = Index(k, "h₂")
  h₃ = Index(k, "h₃")
  s₁ = Index(d, "s₁")
  s₂ = Index(d, "s₂")

  ψ = emptyITensor(l, s₁, s₂, r)
  L = emptyITensor(dag(l), l', h₁)
  H₁ = emptyITensor(dag(s₁), s₁', dag(h₁), h₂)
  H₂ = emptyITensor(dag(s₂), s₂', dag(h₂), h₃)
  R = emptyITensor(dag(r), r', h₃)
  return [ψ, L, H₁, H₂, R]
end

function main()
  mrange = 50:10:80
  krange = 50:10:80
  sequence_costs = Matrix{Any}(undef, length(mrange), length(krange))
  for iₘ in eachindex(mrange), iₖ in eachindex(krange)
    m_val = mrange[iₘ]
    k_val = krange[iₖ]
    d_val = 4

    TN = tensor_network(; m=m_val, k=k_val, d=d_val)
    sequence = optimal_contraction_sequence(TN)
    cost = contraction_cost(TN; sequence=sequence)

    @variables m, k, d
    TN_symbolic = tensor_network(; m=m, k=k, d=d)
    cost_symbolic = contraction_cost(TN_symbolic; sequence=sequence)
    sequence_cost = (
      dims=(m=m_val, k=k_val, d=d_val),
      sequence=sequence,
      cost=cost,
      symbolic_cost=cost_symbolic,
    )
    sequence_costs[iₘ, iₖ] = sequence_cost
  end
  return sequence_costs
end

sequence_costs = main()

# Analyze the results.
println("Index dimensions")
display(getindex.(sequence_costs, :dims))

println("\nContraction sequences")
display(getindex.(sequence_costs, :sequence))

println("\nSymbolic contraction cost with d = 4")
# Fix d to a certain value (such as 4 for a Hubbard site)
@variables d
var_sub = Dict(d => 4)
display(substitute.(sum.(getindex.(sequence_costs, :symbolic_cost)), (var_sub,)))
