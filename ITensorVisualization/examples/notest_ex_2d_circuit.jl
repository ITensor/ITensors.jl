using ITensors
using ITensorsVisualization
using Graphs
using GLMakie
using LayeredLayouts
using PastaQ: randomcircuit

include("utils/circuit_network.jl")
include("utils/layered_layout.jl")

Nx, Ny = 3, 3
N = Nx * Ny
# TODO: change to (Nx, Ny) with PastaQ v0.0.16
gates = randomcircuit(Nx, Ny; depth=4, twoqubitgates="CX", onequbitgates="Rn", layered=false, rotated=false)

s = siteinds("Qubit", N)

U, s̃ = circuit_network(gates, s)
ψ = MPS(s)
ψ̃ = MPS(s̃)
tn = [prod(ψ), U..., prod(ψ̃)]

original_backend = ITensorsVisualization.set_backend!("Makie")

edge_labels = (; plevs=true)
@visualize fig tn arrow_show=true edge_labels=edge_labels layout=layered_layout edge_textsize=20
@visualize! fig[2, 1] tn ndims=3 arrow_show=true edge_labels=edge_labels edge_textsize=10

ITensorsVisualization.set_backend!(original_backend)

fig
