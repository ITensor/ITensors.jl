using ITensors
using ITensorMakie
using Graphs
using GLMakie

include("utils/layered_layout.jl")

tn = itensornetwork(grid((4, 4)); linkspaces=3)
@visualize fig tn arrow_show = true layout = layered_layout

fig
