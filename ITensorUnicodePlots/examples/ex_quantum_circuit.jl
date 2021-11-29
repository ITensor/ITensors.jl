using ITensors
using ITensorUnicodePlots
using Graphs
using LayeredLayouts

N = 6
layers = 6
ndelete = 0

s = siteinds("Qubit", N)
layer(N, start) = [("CX", i, i + 1) for i in start:2:(N - 1)]
layer(N) = append!(layer(N, 1), layer(N, 2))
layer_N = layer(N)
gates = []
for _ in 1:layers
  append!(gates, layer_N)
end

for _ in 1:ndelete
  deleteat!(gates, rand(eachindex(gates)))
end

U, s̃ = circuit_network(gates, s)
ψ = prod(MPS(s))
ψ̃ = prod(MPS(s̃))
tn = [ψ, U..., ψ̃]

edge_labels = (; plevs=true)
layout(g) = layered_layout(solve_positions(Zarate(), g))
@visualize fig tn arrow_show = true edge_labels = edge_labels layout = layout width = 90 height =
  40

fig
