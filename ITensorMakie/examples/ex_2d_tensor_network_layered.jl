using ITensors
using ITensorVisualization
using LayeredLayouts
using Graphs
using GLMakie

function layout(g)
  xs, ys, _ = solve_positions(Zarate(), g)
  return Point.(zip(xs, ys))
end

tn = itensornetwork(grid((4, 4)); linkspaces=3)
@visualize fig tn arrow_show = true layout = layout backend = "Makie"

fig
