module ITensorVisualizationBase

using AbstractTrees
using Compat
using GeometryBasics
using Graphs
using ITensors
using ITensors.ITensorVisualizationCore
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

import ITensors.ITensorVisualizationCore: visualize, visualize!, visualize_sequence

export @visualize,
  @visualize!,
  @visualize_noeval,
  @visualize_noeval!,
  @visualize_sequence,
  @visualize_sequence_noeval,
  circuit_network,
  itensornetwork,
  layered_layout,
  IndexLabels

# Some general graph functionality
include("graphs.jl")

# Some general layout functionality
include("layered_layout.jl")

# Backends interface
include("backends_interface.jl")
include("defaults.jl")

# Conversion betweens graphs and ITensor networks
include("itensor_graph.jl")

# Visualizing ITensor networks
include("visualize.jl")

end
