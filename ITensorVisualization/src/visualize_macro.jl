
#
# Contraction sequence
#

# Tools for contracting a network with a sequence
function _contract(label1::String, label2::String)
  return string("(", label1, "*", label2, ")")
end

function _contract(tensor1::ITensor, tensor2::ITensor)
  return ITensor(noncommoninds(tensor1, tensor2))
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
function visualize(tn::Tuple{Vararg{ITensor}}, args...; kwargs...)
  return visualize(collect(tn), args...; kwargs...)
end
visualize(tn::ITensor...; kwargs...) = visualize(collect(tn); kwargs...)

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

function visualize(f::Union{Function,Type}, As...; kwargs...)
  # TODO: specialize on the function type. Also accept a general collection.
  return visualize(As...; kwargs...)
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

  for n in 1:(length(tn_sequence) - 1)
    visualize!(fig[1, n + 2], tn_sequence[n]; vertex_labels=labels_sequence[n], kwargs...)
  end

  return fig
end

function visualize_sequence(
  f::Union{Function,Type}, tn::Tuple{Vector{ITensor}}, sequence; kwargs...
)
  return visualize_sequence(f, tn[1], sequence; kwargs...)
end

is_kwarg(arg_or_kwarg::Symbol) = false
is_kwarg(arg_or_kwarg::Expr) = (arg_or_kwarg.head == :parameters)

function has_kwargs(args_kwargs::Vector)
  isempty(args_kwargs) && return false
  return is_kwarg(first(args_kwargs))
end

function get_kwargs(args_kwargs::Vector)
  @assert has_kwargs(args_kwargs)
  return first(args_kwargs)
end

function get_kwarg(kwargs::Expr, key::Symbol)
  n = findfirst(kw -> kw.args[1] == :sequence, kwargs.args)
  if !isnothing(n)
    @assert kwargs.args[n].head == :kw
    return esc(kwargs.args[n].args[2])
  end
  return nothing
end

function args_kwargs(ex::Vector)
  kwargs = has_kwargs(ex) ? get_kwargs(ex) : :()
  args = has_kwargs(ex) ? ex[2:end] : ex
  return args, kwargs
end

function function_args_kwargs(ex::Symbol)
  func = :identity
  args = [ex]
  kwargs = :()
  iscollection = true
  return func, args, kwargs, iscollection
end

function function_args_kwargs(ex::Expr)
  if ex.head == :call
    func = first(ex.args)
    args, kwargs = args_kwargs(ex.args[2:end])
    iscollection = true
  elseif ex.head == :ref
    #func, args, kwargs, iscollection = function_args_kwargs(Symbol(ex.args))
    func = :identity
    args = [ex]
    kwargs = :()
    iscollection = false
  else
    dump(ex)
    error("Visualizing expression $ex not supported right now.")
  end
  return func, args, kwargs, iscollection
end

expr_to_string(s::Symbol) = String(s)
expr_to_string(ex::Expr) = String(repr(ex))[3:(end - 1)]

# Take the symbols of the arguments and output
# the labels if there are multiple inputs or
# the prefix for the labels if there is only
# one input.
function vertex_labels_kwargs(args, iscollection)
  if iscollection && isone(length(args))
    vertex_labels_kw = :vertex_labels_prefix
    vertex_labels_arg = string(only(args))
  else
    vertex_labels_kw = :vertex_labels
    vertex_labels_arg = string.(args)
  end
  return vertex_labels_kw, vertex_labels_arg
end

function func_args_sequence_kwargs(ex, vis_kwargs...)
  func, args, kwargs, iscollection = function_args_kwargs(ex)
  sequence = get_kwarg(kwargs, :sequence)
  vertex_labels_kw, vertex_labels_arg = vertex_labels_kwargs(args, iscollection)
  # Merge labels kwarg with kwargs
  vis_kwargs_dict = Dict([
    vis_kwarg.args[1] => vis_kwarg.args[2] for vis_kwarg in vis_kwargs
  ])
  vertex_labels_kwarg_dict = Dict(vertex_labels_kw => vertex_labels_arg)
  merged_kwargs_dict = merge(vertex_labels_kwarg_dict, vis_kwargs_dict)
  merged_kwargs_expr = [:($k = $v) for (k, v) in pairs(merged_kwargs_dict)]
  return func, esc.(args), sequence, esc.(merged_kwargs_expr)
end

function visualize_expr(vis_func, ex::Union{Symbol,Expr}, vis_kwargs::Expr...)
  func, args, sequence, kwargs = func_args_sequence_kwargs(ex, vis_kwargs...)
  e = quote
    $(vis_func)($(func), ($(args...),), $(sequence); $(kwargs...))
  end
  return e
end

function visualize_expr!(fig, vis_func!, ex::Union{Symbol,Expr}, vis_kwargs::Expr...)
  func, args, sequence, kwargs = func_args_sequence_kwargs(ex, vis_kwargs...)
  e = quote
    $(vis_func!)($(esc(fig)), $(func), ($(args...),), $(sequence); $(kwargs...))
  end
  return e
end

"""
    @visualize

Visualize a contraction of ITensors, returning the result of the contraction.

The contraction should be written in terms of a series of ITensors contracted with `*`.

# Examples
```julia
using ITensors
using ITensorVisualization

i = Index(2, "index_i")
j = Index(10, "index_j")
k = Index(40, "index_k")
l = Index(40, "index_l")
m = Index(40, "index_m")
A = randomITensor(i, j, k)
B = randomITensor(i, j, l, m)
C = randomITensor(k, l)

# Contract the tensors over the common indices
# and visualize the results
ABC = @visualize A * B * C

AB = @visualize A * B
# Use readline() to pause between plots
readline()
ABC = @visualize AB * C vertex_labels = ["A*B", "C"]
readline()

# Save the results to figures for viewing later
AB = @visualize fig1 A * B
ABC = @visualize fig2 AB * C vertex_labels = ["A*B", "C"]

display(fig1)
readline()
display(fig2)
readline()
```

# Keyword arguments:
- `vertex_labels`: Custom tensor labels to display on the vertices of the digram. If not specified, they are determined automatically from the input to the macro.
- `edge_labels=IndexLabels()`: A list of the edge labels or an `AbstractEdgeLabels` object specifying how they should be made.
- `arrow_show`: Whether or not to show arrows on the edges.
"""
macro visualize(fig::Symbol, ex::Symbol, kwargs::Expr...)
  e = quote
    $(esc(fig)) = $(visualize_expr(visualize, ex, kwargs...))
    $(esc(ex))
  end
  return e
end

macro visualize!(fig, ex::Symbol, kwargs::Expr...)
  e = quote
    $(visualize_expr!(fig, visualize!, ex, kwargs...))
    $(esc(ex))
  end
  return e
end

macro visualize(ex::Symbol)
  e = quote
    display($(visualize_expr(visualize, ex)))
    $(esc(ex))
  end
  return e
end

macro visualize(ex_or_fig::Symbol, ex_or_kwarg::Expr, last_kwargs::Expr...)
  if ex_or_kwarg.head == :(=)
    # The second input is a keyword argument which means that the
    # first input is the collection to visualize (no figure output binding specified)
    ex = ex_or_fig
    kwargs = (ex_or_kwarg, last_kwargs...)
    e = quote
      display($(visualize_expr(visualize, ex, kwargs...)))
      $(esc(ex))
    end
  else
    # The second input is not a keyword argument which means that the
    # first input is the binding for the figure output, the second is the expression
    # to visualize
    fig = ex_or_fig
    ex = ex_or_kwarg
    kwargs = last_kwargs
    e = quote
      $(esc(fig)) = $(visualize_expr(visualize, ex, kwargs...))
      $(esc(ex))
    end
  end
  return e
end

macro visualize!(fig, ex::Expr, kwargs::Expr...)
  e = quote
    $(visualize_expr!(fig, visualize!, ex, kwargs...))
    $(esc(ex))
  end
  return e
end

macro visualize(ex::Expr, kwargs::Expr...)
  e = quote
    display($(visualize_expr(visualize, ex, kwargs...)))
    $(esc(ex))
  end
  return e
end

macro visualize_noeval(ex::Symbol, kwargs::Expr...)
  e = quote
    $(visualize_expr(visualize, ex, kwargs...))
  end
  return e
end

macro visualize_noeval(ex::Expr, kwargs::Expr...)
  e = quote
    $(visualize_expr(visualize, ex, kwargs...))
  end
  return e
end

macro visualize_noeval!(fig, ex::Symbol, kwargs::Expr...)
  e = quote
    $(visualize_expr!(fig, visualize!, ex, kwargs...))
  end
  return e
end

macro visualize_noeval!(fig, ex::Expr, kwargs::Expr...)
  e = quote
    $(visualize_expr!(fig, visualize!, ex, kwargs...))
  end
  return e
end

macro visualize_sequence(fig::Symbol, ex::Expr, kwargs::Expr...)
  e = quote
    $(esc(fig)) = $(visualize_expr(visualize_sequence, ex, kwargs...))
    $(esc(ex))
  end
  return e
end

macro visualize_sequence(ex::Expr, kwargs::Expr...)
  e = quote
    display($(visualize_expr(visualize_sequence, ex, kwargs...)))
    $(esc(ex))
  end
  return e
end

macro visualize_sequence_noeval(ex::Expr, kwargs::Expr...)
  e = quote
    $(visualize_expr(visualize_sequence, ex, kwargs...))
  end
  return e
end
