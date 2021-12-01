using ITensors
using ITensorVisualizationBase
using LayeredLayouts
using Graphs

tn = itensornetwork(grid((4, 4)); linkspaces=3)
layout(g) = layered_layout(solve_positions(Zarate(), g))
@visualize fig tn arrow_show = true layout = layout

fig
