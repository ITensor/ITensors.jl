using ITensors
using ITensorsVisualization
using LayeredLayouts
using Graphs
using GLMakie

include("utils/circuit_network.jl")
include("utils/layered_layout.jl")

N = 10
layers = 10
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
@visualize fig tn arrow_show=true edge_labels=edge_labels layout=layered_layout backend="Makie"

fig
