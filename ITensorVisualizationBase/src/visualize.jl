#
# Contraction sequence
#

# Tools for contracting a network with a sequence
function _contract(label1::String, label2::String)
  return string("(", label1, "*", label2, ")")
end

function _contract(tensor1::ITensor, tensor2::ITensor)
  indsR = noncommoninds(tensor1, tensor2)
  return isempty(indsR) ? ITensor() : ITensor(indsR)
end

sequence_traversal(sequence) = reverse(collect(StatelessBFS(sequence)))

function contract_dict(tensors, sequence, traversal=sequence_traversal(sequence))
  net_tensors = Dict()
  traversal = reverse(collect(StatelessBFS(sequence)))
  for net in traversal
    if net isa Int
      net_tensors[net] = tensors[net]
    else # net isa Vector
      net_tensors[net] = _contract(net_tensors[net[1]], net_tensors[net[2]])
    end
  end
  return net_tensors
end

# Return all of the contractions involved in the sequence.
function contraction_sequence(
  tensors,
  sequence,
  traversal=sequence_traversal(sequence),
  contract_dict=contract_dict(tensors, sequence, traversal),
)
  all_tensors = Any[]
  tensors_1 = Vector{Union{Nothing,eltype(tensors)}}(tensors)
  net_position = Dict()
  N = length(tensors)
  n = N + 1
  for net in traversal
    if net isa Int
      net_position[net] = net
    else
      net_position[net] = n
      n += 1
    end
    if !isa(net, Int)
      for n in net
        tensors_1[net_position[n]] = nothing
      end
      push!(tensors_1, contract_dict[net])
      push!(all_tensors, copy(tensors_1))
    end
  end
  return convert.(Vector{eltype(tensors)}, filter.(!isnothing, all_tensors))
end

#
# Convert a tree to a graph
#

struct Tree
  x::Any
end
function Base.getindex(tree::Tree, indices)
  node = tree.x
  for idx in indices
    node = children(node)[idx]
  end
  return node
end

tree_to_graph(tr) = tree_to_graph(Tree(tr))

function tree_to_graph(tr::Tree)
  g = SimpleDiGraph()
  labels = Any[]
  walk_tree!(g, labels, tr)
  return (g, labels)
end

function walk_tree!(g, labels, tr::Tree)
  add_vertex!(g)
  top_vertex = vertices(g)[end]
  push!(labels, tr.x)
  for i in 1:length(tr.x)
    if isa(tr[i], Vector)
      child = walk_tree!(g, labels, Tree(tr[i]))
      add_edge!(g, child, top_vertex)
    else
      add_vertex!(g)
      n = vertices(g)[end]
      add_edge!(g, n, top_vertex)
      push!(labels, tr[i])
    end
  end
  return top_vertex
end

# Visualization function interface. Ultimately calls a beckend.

function visualize(g::AbstractGraph, sequence=nothing; backend=get_backend(), kwargs...)
  # TODO: do something with the sequence (show sequence, add labels indicating sequence, etc.)
  return visualize(Backend(backend), g; kwargs...)
end

function visualize(tn::Vector{ITensor}, sequence=nothing; kwargs...)
  return visualize(MetaDiGraph(tn), sequence; kwargs...)
end

function visualize(tn::Tuple{Vector{ITensor}}, args...; kwargs...)
  return visualize(only(tn), args...; kwargs...)
end
visualize(ψ::MPS, args...; kwargs...) = visualize(data(ψ), args...; kwargs...)
function visualize(tn::Tuple{ITensor,Vararg{ITensor}}, args...; kwargs...)
  return visualize(collect(tn), args...; kwargs...)
end
function visualize(t1::ITensor, tn_tail::ITensor...; kwargs...)
  return visualize([t1, tn_tail...]; kwargs...)
end

# Special case single ITensor
function visualize(t::ITensor, sequence=nothing; vertex_labels_prefix, kwargs...)
  tn = [t]
  vertex_labels = [vertex_labels_prefix]
  return visualize(MetaDiGraph(tn), sequence; vertex_labels=vertex_labels, kwargs...)
end

# Special case single ITensor
function visualize(tn::Tuple{ITensor}, args...; kwargs...)
  return visualize(only(tn), args...; kwargs...)
end

function visualize!(fig, g::AbstractGraph; backend=get_backend(), kwargs...)
  return visualize!(Backend(backend), fig, g; kwargs...)
end

function visualize!(fig, tn::Vector{ITensor}, sequence=nothing; kwargs...)
  return visualize!(fig, MetaDiGraph(tn); kwargs...)
end
visualize!(fig, ψ::MPS, sequence=nothing; kwargs...) = visualize!(fig, data(ψ); kwargs...)
function visualize!(fig, tn::Tuple{Vararg{ITensor}}, sequence=nothing; kwargs...)
  return visualize!(fig, collect(tn); kwargs...)
end
visualize!(fig, tn::ITensor...; kwargs...) = visualize!(fig, collect(tn); kwargs...)

function visualize!(fig, tn::Tuple{Vector{ITensor}}, sequence=nothing; kwargs...)
  return visualize!(fig, tn[1], sequence; kwargs...)
end
function visualize!(
  fig, f::Function, tn::Tuple{Vararg{ITensor}}, sequence=nothing; kwargs...
)
  return visualize!(fig, tn, sequence; kwargs...)
end

# Macro outputs a 1-tuple of the function arguments
function visualize(f::Union{Function,Type}, tn::Tuple{T}, sequence; kwargs...) where {T}
  # TODO: specialize on the function type. Also accept a general collection.
  return visualize(only(tn), sequence; kwargs...)
end

# Macro outputs a tuple of ITensors to visualize
function visualize(f::Union{Function,Type}, tn::Tuple{Vararg{ITensor}}, sequence; kwargs...)
  # TODO: specialize on the function type. Also accept a general collection.
  return visualize(tn, sequence; kwargs...)
end

function visualize!(fig, f::Union{Function,Type}, As...; kwargs...)
  # TODO: specialize of the function type. Also accept a general collection.
  return visualize!(fig, As...; kwargs...)
end

function _visualize_sequence!(fig, tn, sequence, n; kwargs...)
  return error("Not implemented")
end

function sequence_labels(sequence, all_sequences, vertex_labels)
  traversal = sequence_traversal(sequence)
  labels_dict = contract_dict(vertex_labels, sequence, traversal)
  all_labels = [labels_dict[s] for s in all_sequences]
  return all_labels
end

function _graphplot(backend::Backend, graph; all_labels)
  return error("Not implemented for backend $backend.")
end

function visualize_sequence(sequence, vertex_labels)
  graph, all_sequences = tree_to_graph(sequence)
  all_labels = sequence_labels(sequence, all_sequences, vertex_labels)
  fig = _graphplot(Backend"Makie"(), graph; all_labels=all_labels)
  return fig
end

function default_sequence(tn::Vector{ITensor})
  N = length(tn)
  return foldl((x, y) -> [x, y], 1:N)
end

function visualize_sequence(
  f::Union{Function,Type}, tn::Vector{ITensor}, sequence::Nothing; kwargs...
)
  return visualize_sequence(f, tn, default_sequence(tn); kwargs...)
end

function visualize_sequence(
  f::Union{Function,Type}, tn::Vector{ITensor}, sequence=default_sequence(tn); kwargs...
)
  N = length(tn)

  # TODO: clean this up a bit
  vertex_labels_prefix = get(
    kwargs,
    :vertex_labels_prefix,
    default_vertex_labels_prefix(Backend("Makie"), MetaDiGraph(tn)),
  )
  vertex_labels = get(
    kwargs,
    :vertex_labels,
    default_vertex_labels(Backend(""), MetaDiGraph(tn), vertex_labels_prefix),
  )

  fig = visualize_sequence(sequence, vertex_labels)

  visualize!(fig[1, 2], tn; vertex_labels=vertex_labels, kwargs...)

  traversal = sequence_traversal(sequence)
  labels_sequence = contraction_sequence(vertex_labels, sequence, traversal)

  tn_sequence = contraction_sequence(tn, sequence, traversal)

  for n in 1:length(tn_sequence)
    visualize!(fig[1, n + 2], tn_sequence[n]; vertex_labels=labels_sequence[n], kwargs...)
  end

  return fig
end

function visualize_sequence(
  f::Union{Function,Type}, tn::Tuple{Vector{ITensor}}, sequence; kwargs...
)
  return visualize_sequence(f, tn[1], sequence; kwargs...)
end
