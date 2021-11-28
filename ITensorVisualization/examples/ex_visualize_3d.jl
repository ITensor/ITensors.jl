using ITensors
using ITensorVisualization
using GLMakie
using Graphs

tn = itensornetwork(grid((3, 3, 3)))
edge_labels = (; dims=false)
@visualize fig tn ndims = 3 edge_labels = edge_labels vertex_size = 400 backend = "Makie"

fig
