using ITensors
using ITensorGLMakie
using Graphs

using ITensors.ContractionSequenceOptimization: optimal_contraction_sequence

N = 5
g = Graph(N)
g_edges = [2 => 3, 1 => 4, 1 => 5, 4 => 5]
for e in g_edges
  add_edge!(g, e)
end

A = itensornetwork(g; linkspaces=5)
sequence = optimal_contraction_sequence(A)
edge_labels = (; tags=true)
R = @visualize_sequence fig ITensors.contract(A; sequence=sequence) edge_labels =
  edge_labels

fig
