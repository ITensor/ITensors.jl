visualize(args...; kwargs...) = nothing
visualize!(args...; kwargs...) = nothing
visualize_sequence(args...; kwargs...) = nothing

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
using ITensorUnicodePlots # Must load a backend or else no plots will be made

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

  - `vertex_labels`: Custom tensor labels to display on the vertices of the
     digram. If not specified, they are determined automatically from the input to the macro.
  - `edge_labels=IndexLabels()`: A list of the edge labels or an
     `AbstractEdgeLabels` object specifying how they should be made.
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
