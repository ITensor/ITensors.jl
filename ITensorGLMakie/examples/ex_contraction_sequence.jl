using ITensors
using ITensorGLMakie
using Graphs: Graph
using Random

using ITensors.ContractionSequenceOptimization: optimal_contraction_sequence

Random.seed!(1234)

N = 5
g = Graph(N, N)
A = itensornetwork(g; linkspaces=5)
sequence = optimal_contraction_sequence(A)
edge_labels = (; tags=true)
R = @visualize_sequence fig ITensors.contract(A; sequence=sequence) edge_labels =
  edge_labels

fig
