using ITensors
using ITensorUnicodePlots
using Graphs
using PastaQ: qft

include("utils/layered_layout.jl")
include("utils/circuit_network.jl")

N = 4
gates = qft(N)

s = siteinds("Qubit", N)

U, s̃ = circuit_network(gates, s)
ψ = MPS(s)
ψ̃ = MPS(s̃)
tn = [ψ..., U..., ψ̃...]

edge_labels = (; tags=true, plevs=true)
@visualize fig tn arrow_show = true edge_labels = edge_labels edge_textsize = 20 layout =
  layered_layout width = 100 height = 50

fig
