using ITensors
using ITensorsVisualization
using Graphs
using GLMakie
using LayeredLayouts
using PastaQ: qft

include("utils/circuit_network.jl")
include("utils/layered_layout.jl")

N = 4
gates = qft(N)

s = siteinds("Qubit", N)

U, s̃ = circuit_network(gates, s)
ψ = MPS(s)
ψ̃ = MPS(s̃)
tn = [ψ..., U..., ψ̃...]

original_backend = ITensorsVisualization.set_backend!("Makie")

edge_labels = (; tags=true, plevs=true)
@visualize fig tn arrow_show=true edge_labels=edge_labels edge_textsize=20 layout=layered_layout
edge_labels = (; plevs=true)
@visualize! fig[1, 2] tn ndims=3 edge_labels=edge_labels edge_textsize=20

ITensorsVisualization.set_backend!(original_backend)

fig
