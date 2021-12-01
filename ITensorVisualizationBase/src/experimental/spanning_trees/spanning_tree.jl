using Graphs, SimpleWeightedGraphs
using GraphRecipes, Plots

#spanning_tree_method = "mst"
spanning_tree_method = "bfs"

nx, ny = 7, 7
n = nx * ny
g = Graphs.grid((nx, ny); periodic=false)

nx_middle = 1 #nx ÷ 2 + 1
ny_middle = 1 #ny ÷ 2 + 1
n_middle = LinearIndices((nx, ny))[nx_middle, ny_middle]

vert_names = string.(vertices(g))
dist = gdistances(g, n_middle)
#names = string.(vert_names, ", ", dist)
names = dist

wg = SimpleWeightedGraph(nv(g))

for e in edges(g)
  dw = mean([dist[src(e)], dist[dst(e)]])
  w = 1 / dw^2 + eps() * randn()
  add_edge!(wg, src(e), dst(e), w)
end

g_st = if spanning_tree_method == "mst"
  # mst_function = boruvka_mst
  mst_function = kruskal_mst
  mst_weight = mst_function(wg; minimize=false)
  mst = mst_function == boruvka_mst ? mst_weight.mst : mst_weight
  g_mst = SimpleWeightedGraph(nv(wg))
  for ew in mst
    add_edge!(g_mst, src(ew), dst(ew), weight(ew))
  end
  g_mst
elseif spanning_tree_method == "bfs"
  # Weights are set to 1
  SimpleWeightedGraph(bfs_tree(wg, n_middle))
end

edgelabel_dict = Dict{Tuple{Int,Int},String}()
for ew in edges(wg)
  edgelabel_dict[(src(ew), dst(ew))] = string(round(weight(ew); digits=2))
end

edgecolor_dict = Dict()
for ew in edges(wg)
  color = ew ∈ edges(g_st) ? :black : :red
  edgecolor_dict[(src(ew), dst(ew))] = color
end

edgelabel_dict_mst = Dict()
for i in vertices(g_st), j in vertices(g_st)
  edgelabel_dict_mst[(i, j)] = string(round(get_weight(g_st, i, j); digits=2))
end

plt = graphplot(
  wg;
  markersize=0.3,
  names=names,
  edgelabel=edgelabel_dict,
  curves=false,
  edgecolor=edgecolor_dict,
  linewidth=20,
  fontsize=20,
  size=(3000, 3000),
)

plt
