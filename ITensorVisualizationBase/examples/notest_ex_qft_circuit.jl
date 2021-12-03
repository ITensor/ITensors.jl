using ITensors
using ITensorVisualizationBase
using Graphs
using LayeredLayouts
using PastaQ: qft

N = 4
gates = qft(N)

s = siteinds("Qubit", N)

U, s̃ = circuit_network(gates, s)
ψ = MPS(s)
ψ̃ = MPS(s̃)
tn = [ψ..., U..., ψ̃...]

edge_labels = (; tags=true, plevs=true)
layout(g) = layered_layout(solve_positions(Zarate(), g))
@visualize fig tn arrow_show = true edge_labels = edge_labels edge_textsize = 20 layout =
  layout

fig
