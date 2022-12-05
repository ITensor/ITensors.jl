module ITensorMakie

using Colors
using Graphs
using NetworkLayout
using Reexport
using GraphMakie
@reexport using ITensorVisualizationBase

using GraphMakie.Makie:
  Makie,
  Figure,
  contents,
  hidedecorations!,
  hidespines!,
  deregister_interaction!,
  register_interaction!

using ITensorVisualizationBase:
  @Backend_str,
  default_vertex_labels,
  default_vertex_labels_prefix,
  default_vertex_size,
  default_vertex_textsize,
  default_edge_textsize,
  default_edge_widths,
  default_edge_labels,
  default_arrow_show,
  default_arrow_size,
  default_siteinds_direction,
  is_self_loop,
  _ndims

import ITensorVisualizationBase: visualize, visualize!, _graphplot

function __init__()
  return ITensorVisualizationBase.set_backend!(Backend"Makie"())
end

fill_number(a::AbstractVector, n::Integer) = a
fill_number(x::Number, n::Integer) = fill(x, n)

function visualize(b::Backend"Makie", g::AbstractGraph; kwargs...)
  f = Figure()
  visualize!(b, f[1, 1], g; kwargs...)
  return f
end

function visualize!(b::Backend"Makie", f::Figure, g::AbstractGraph; kwargs...)
  visualize!(b, f[1, 1], g; kwargs...)
  return f
end

function visualize!(
  b::Backend"Makie",
  f::Makie.GridPosition,
  g::AbstractGraph;
  interactive=true,
  ndims=2,
  layout=Spring(; dim=ndims),

  # vertex
  vertex_labels_prefix=default_vertex_labels_prefix(b, g),
  vertex_labels=default_vertex_labels(b, g, vertex_labels_prefix),
  vertex_size=default_vertex_size(b, g),
  vertex_textsize=default_vertex_textsize(b, g),

  # edge
  edge_textsize=default_edge_textsize(b),
  edge_widths=default_edge_widths(b, g),
  edge_labels=default_edge_labels(b, g),

  # arrow
  arrow_show=default_arrow_show(b, g),
  arrow_size=default_arrow_size(b, g),
  siteinds_direction=default_siteinds_direction(b, g),
)
  if ismissing(Makie.current_backend())
    error(
      """
  You have not loaded a backend.  Please load one (`using GLMakie` or `using CairoMakie`)
  before trying to visualize a graph.
"""
    )
  end

  edge_labels = ITensorVisualizationBase.edge_labels(b, edge_labels, g)

  if length(vertex_labels) ≠ nv(g)
    throw(
      DimensionMismatch(
        "$(length(vertex_labels)) vertex labels $(vertex_labels) were specified
        but there are $(nv(g)) tensors in the diagram, please specify the
        correct number of labels."
      ),
    )
  end

  graphplot_kwargs = (;
    layout=layout,

    # vertex
    node_size=fill_number(vertex_size, nv(g)),
    node_color=colorant"lightblue1", # TODO: store in vertex, make a default
    node_marker='●', # TODO: allow other options, like '◼'
    node_attr=(; strokecolor=:black, strokewidth=3),

    # vertex labels
    nlabels=vertex_labels,
    nlabels_textsize=vertex_textsize,
    nlabels_color=colorant"black",
    nlabels_align=(:center, :center),

    # edge
    edge_width=edge_widths,
    edge_color=colorant"black",

    # edge labels
    elabels=edge_labels,
    elabels_textsize=edge_textsize,
    elabels_color=colorant"red",

    # self-edge
    selfedge_width=1e-5, # Small enough so you can't see the loop, big enough for site label to show up
    selfedge_direction=siteinds_direction,
    selfedge_size=3,

    # arrow
    arrow_show=arrow_show,
    arrow_size=arrow_size,
    arrow_shift=0.49,
  )

  overwrite_axis = false
  if isempty(contents(f))
    axis_plot = graphplot(f, g; graphplot_kwargs...)
  else
    @warn "Visualizing a graph in the same axis as an existing graph. This
    feature is experimental and some features like interactivity might now work"
    overwrite_axis = true
    graphplot!(f, g; graphplot_kwargs...)
  end

  # Set rotation of edge labels to 0.
  axis_plot.plot.elabels_rotation[] = zeros(ne(g))

  if !overwrite_axis && (_ndims(layout) == 2)
    hidedecorations!(axis_plot.axis)
    # This would hide the box around the plot
    # TODO: make this optional
    #hidespines!(axis_plot.axis)
    if interactive
      deregister_interaction!(axis_plot.axis, :rectanglezoom)
      register_interaction!(axis_plot.axis, :nhover, NodeHoverHighlight(axis_plot.plot))
      register_interaction!(axis_plot.axis, :ehover, EdgeHoverHighlight(axis_plot.plot))
      register_interaction!(axis_plot.axis, :ndrag, NodeDrag(axis_plot.plot))
      register_interaction!(axis_plot.axis, :edrag, EdgeDrag(axis_plot.plot))
    end
  end
  return f
end

# For use in sequence visualization.
# TODO: Make this more generalizable to other backends.
function _graphplot(::Backend"Makie", graph; all_labels)
  fig, ax, plt = graphplot(
    reverse(graph); arrow_show=false, nlabels=all_labels, layout=Buchheim()
  )
  hidedecorations!(ax)
  #hidespines!(ax)
  return fig
end

end
