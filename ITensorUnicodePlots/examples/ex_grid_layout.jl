using ITensors
using ITensorUnicodePlots
using Graphs
using GeometryBasics
using NetworkLayout

N = 10
g = grid((N,))
tn = itensornetwork(g; linkspaces=10, sitespaces=2)
@visualize fig tn siteinds_direction = Point(1, -0.5) layout = SquareGrid(; cols=1) width =
  20 height = 50

fig
