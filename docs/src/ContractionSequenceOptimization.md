# Contraction sequence optimization

When contracting a tensor network, the sequence of contraction makes a big difference in the computational cost. However, the complexity of determining the optimal sequence grows exponentially with the number of tensors, but there are many heuristic algorithms available for computing optimal sequences for small networks[^1][^2][^3][^4][^5][^6]. ITensors.jl provides some functionality for helping you find the optimal contraction sequence for small tensor network, as we will show below.

The algorithm in ITensors.jl currently uses a modified version of[^1] with simplifications for outer product contractions similar to those used in [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl).

[^1]: [Faster identification of optimal contraction sequences for tensor networks](https://arxiv.org/abs/1304.6112)
[^2]: [Improving the efficiency of variational tensor network algorithms](https://arxiv.org/abs/1310.8023)
[^3]: [Simulating quantum computation by contracting tensor networks](https://arxiv.org/abs/quant-ph/0511069)
[^4]: [Towards a polynomial algorithm for optimal contraction sequence of tensor networks from trees](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.043309)
[^5]: [Algorithms for Tensor Network Contraction Ordering](https://arxiv.org/abs/2001.08063)
[^6]: [Hyper-optimized tensor network contraction](https://arxiv.org/abs/2002.01935)

## Functions

```@docs
ITensors.optimal_contraction_sequence
ITensors.contraction_cost
contract
```

## Examples

In the following example we show how to compute the contraction sequence cost of a 
```@julia
using ITensors
using Symbolics

using ITensors: contraction_cost

@variables m, k, d

l = Index(m, "l")
r = Index(m, "r")
h₁ = Index(k, "h₁")
h₂ = Index(k, "h₂")
h₃ = Index(k, "h₃")
s₁ = Index(d, "s₁")
s₂ = Index(d, "s₂")

H₁ = emptyITensor(dag(s₁), s₁', dag(h₁), h₂)
H₂ = emptyITensor(dag(s₂), s₂', dag(h₂), h₃)
L = emptyITensor(dag(l), l', h₁)
R = emptyITensor(dag(r), r', h₃)
ψ = emptyITensor(l, s₁, s₂, r)

TN = [ψ, L, H₁, H₂, R]
sequence1 = Any[2, Any[3, Any[4, Any[1, 5]]]]
sequence2 = Any[Any[4, 5], Any[1, Any[2, 3]]]
cost1 = contraction_cost(TN; sequence = sequence1)
cost2 = contraction_cost(TN; sequence = sequence2)

println("First sequence")
display(sequence1)
display(cost1)
@show sum(cost1)
@show substitute(sum(cost1), Dict(d => 4))

println("\nSecond sequence")
display(sequence2)
display(cost2)
@show sum(cost2)
@show substitute(sum(cost2), Dict(d => 4))
```
This example helps us learn that in the limit of large MPS bond dimension `m`, the first contraction sequence is faster, while in the limit of large MPO bond dimension `k`, the second sequence is faster. This has practical implications for writing an efficient DMRG algorithm in both limits, which we plan to incorporate into ITensors.jl.

Here is a more systematic example of searching through the parameter space to find optimal contraction sequences:
```julia
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

    TN = tensor_network(; m = m_val, k = k_val, d = d_val)
    sequence = optimal_contraction_sequence(TN)
    cost = contraction_cost(TN; sequence = sequence)

    @variables m, k, d
    TN_symbolic = tensor_network(; m = m, k = k, d = d)
    cost_symbolic = contraction_cost(TN_symbolic; sequence = sequence)
    sequence_cost = (dims = (m = m_val, k = k_val, d = d_val), sequence = sequence, cost = cost, symbolic_cost = cost_symbolic)
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
```

A future direction will be to allow optimizing over contraction sequences with the dimensions specified symbolically, so that the optimal sequence in limits of certain dimensions can be found. In addition, we plan to implement more algorithms that work for larger networks, as well as algorithms like[^2] which take an optimal sequence for a closed network and generate optimal sequences for environments of each tensor in the network, which is helpful for computing gradients of tensor networks.

