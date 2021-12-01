using ITensors
using ITensorVisualizationBase
using Graphs
using LayeredLayouts
using PastaQ: randomcircuit

Nx, Ny = 3, 3
N = Nx * Ny
# TODO: change to (Nx, Ny) with PastaQ v0.0.16
gates = randomcircuit(
  Nx, Ny; depth=4, twoqubitgates="CX", onequbitgates="Rn", layered=false, rotated=false
)

s = siteinds("Qubit", N)

U, s̃ = circuit_network(gates, s)
ψ = MPS(s)
ψ̃ = MPS(s̃)
tn = [prod(ψ), U..., prod(ψ̃)]

edge_labels = (; plevs=true)
layout(g) = layered_layout(solve_positions(Zarate(), g))
@visualize fig tn arrow_show = true edge_labels = edge_labels layout = layout edge_textsize =
  20

fig
