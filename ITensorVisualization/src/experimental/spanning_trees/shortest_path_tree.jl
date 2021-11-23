using Graphs
using Random

Random.seed!(1234)

function dijkstra_spt(g, v_src_initial)
  out = dijkstra_shortest_paths(g, [v_src_initial]; allpaths=true, trackvertices=true)
  paths = out.predecessors
  edges = Edge{Int}[]
  for v_dst_final in eachindex(paths)
    v_src = v_src_initial
    p = paths[v_dst_final]
    @show v_src_initial, p, v_dst_final
    for v_src in p
      push!(edges, Edge(v_src => v_dst_final))
    end
  end
  return edges
end

g = Graph(6, 10)

@show collect(edges(g))

t_mst = kruskal_mst(g)

@show t_mst

t_bfs = bfs_tree(g, 1)

@show collect(edges(t_bfs))

t_spt = dijkstra_spt(g, 1)

@show t_spt
