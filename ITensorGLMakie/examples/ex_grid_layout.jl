using ITensors
using ITensorGLMakie
using Graphs
using NetworkLayout

N = 10
g = grid((N,))
tn = itensornetwork(g; linkspaces=10, sitespaces=2)
@visualize fig tn siteinds_direction = Point(1, -0.5) layout = SquareGrid(; cols=1)

fig
