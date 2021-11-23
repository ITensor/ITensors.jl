module ITensorVisualization

using AbstractTrees
using GeometryBasics
using Graphs
using ITensors
using LinearAlgebra
using MetaGraphs
using NetworkLayout
using SparseArrays
using Statistics

# Avoid conflict between `Graphs.contract` and `ITensors.contract`
using Graphs:
  Graphs,
  AbstractEdge,
  AbstractGraph,
  SimpleGraph,
  SimpleDiGraph,
  add_edge!,
  add_vertex!,
  all_neighbors,
  dst,
  edges,
  ne,
  neighbors,
  nv,
  src,
  vertices

using ITensors: data, QNIndex

export @visualize,
  @visualize!,
  @visualize_noeval,
  @visualize_noeval!,
  @visualize_sequence,
  @visualize_sequence_noeval,
  itensornetwork,
  IndexLabels

# Some general graph functionality
include("graphs.jl")

# Backends interface
include("backends_interface.jl")
include("defaults.jl")

# Conversion betweens graphs and ITensor networks
include("itensor_graph.jl")

# Visualizing ITensor networks
include("visualize_macro.jl")

end
