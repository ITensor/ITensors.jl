#
# Conversion between Graphs and ITensor networks
#

hasuniqueinds(args...; kwargs...) = !isempty(uniqueinds(args...; kwargs...))

function graph_dir(inds)
  dirs = dir.(inds)
  if length(dirs) == 1
    return only(dirs)
  end
  if all(==(dirs[1]), dirs)
    return dirs[1]
  end
  return ITensors.Out
end

# TODO: rename graph, dispatch on QNs to DiGraph
function Graphs.SimpleDiGraph(tn::Vector{ITensor})
  nv = length(tn)
  g = SimpleDiGraph(nv)
  for v1 in 1:nv, v2 in (v1 + 1):nv
    indsᵛ¹ᵛ² = commoninds(tn[v1], tn[v2])
    if !isempty(commoninds(tn[v1], tn[v2]))
      e = v1 => v2
      if graph_dir(indsᵛ¹ᵛ²) == ITensors.In
        e = reverse(e)
      end
      add_edge!(g, e)
    end
  end
  for v in vertices(g)
    if hasuniqueinds(tn[v], tn[all_neighbors(g, v)]...)
      # Add a self-loop
      add_edge!(g, v => v)
    end
  end
  return g
end

# TODO: rename indsgraph, dispatch on QNs to DiGraph
function MetaGraphs.MetaDiGraph(tn::Vector{ITensor})
  sg = SimpleDiGraph(tn)
  mg = MetaDiGraph(sg)
  for e in edges(mg)
    indsₑ = if is_self_loop(e)
      v = src(e)
      # For self edges, the vertex itself is included as
      # a neighbor so we must exclude it.
      uniqueinds(tn[v], tn[setdiff(all_neighbors(mg, v), v)]...)
    else
      commoninds(tn[src(e)], tn[dst(e)])
    end
    set_prop!(mg, e, :inds, indsₑ)
  end
  return mg
end

default_linkspaces() = 1
default_sitespaces() = 1

default(x, x_default) = x
default(x::Nothing, x_default) = x_default

function itensornetwork(
  g::AbstractGraph; linkspaces=default_linkspaces(), sitespaces=nothing
)
  N = nv(g)
  if !isnothing(sitespaces) && !any_self_loops(g)
    g = copy(g)
    for v in vertices(g)
      add_edge!(g, v => v)
    end
  end
  sitespaces = default(sitespaces, default_sitespaces())
  # TODO: Specialize to Index{typeof(linkspaces)}
  inds_network = [Index[] for _ in 1:N]
  for e in edges(g)
    if !is_self_loop(e)
      lₑ = Index(linkspaces; tags="l=$(src(e))↔$(dst(e))")
      push!(inds_network[src(e)], lₑ)
      push!(inds_network[dst(e)], dag(lₑ))
    else
      sₑ = Index(sitespaces; tags="s=$(src(e))")
      push!(inds_network[src(e)], sₑ)
    end
  end
  tn = Vector{ITensor}(undef, N)
  for n in 1:N
    tn[n] = ITensor(inds_network[n])
  end
  return tn
end

sites(g::Tuple{String,<:Tuple}) = g[2]
sites(g::Tuple{String,<:Tuple,<:NamedTuple}) = g[2]
sites(g::Tuple{String,Int}) = g[2]
sites(g::Tuple{String,Vararg{Int}}) = Base.tail(g)
sites(g::Tuple{String,Int,<:NamedTuple}) = g[2]

# Functionality for turning a list of gates into an ITensor
# network.
function circuit_network(gates, s::Vector{<:Index})
  s = copy(s)
  U = ITensor[]
  for g in gates
    push!(U, op(g, s))
    for n in sites(g)
      s[n] = s[n]'
    end
  end
  return U, s
end
