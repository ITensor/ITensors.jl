using ITensors
using ITensorsVisualization
using Graphs
using NetworkLayout

N = 10
g = Grid((N,))
tn = itensornetwork(g; linkspaces=10, sitespaces=2)
@visualize fig tn siteinds_direction=Point(1, -0.5) layout=SquareGrid(; cols=1) backend="UnicodePlots" width=20 height=50
fig
