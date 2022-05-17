#############################################################################
# backend
#

default_backend() = Backend(nothing)

#############################################################################
# vertex labels
#

function subscript_char(n::Integer)
  @assert 0 ≤ n ≤ 9
  return Char(0x2080 + n)
end

function subscript(n::Integer)
  ss = prod(Iterators.reverse((subscript_char(d) for d in digits(abs(n)))))
  if n < 0
    ss = "₋" * ss
  end
  return ss
end

subscript(n) = string(n)

default_vertex_labels_prefix(b::Backend, g) = "T"
function default_vertex_labels(
  b::Backend, g::AbstractGraph, vertex_labels_prefix=default_vertex_labels_prefix(b)
)
  return [string(vertex_labels_prefix, subscript(v)) for v in vertices(g)]
end

default_vertex_size(b::Backend, g) = 60
default_vertex_textsize(b::Backend, g) = 20

# TODO: customizable vertex marker
# nodeshapes="●", # ●, ▶, ◀, ■, █, ◩, ◪, ⧄, ⧅, ⦸, ⊘, ⬔, ⬕, ⬛, ⬤, 🔲, 🔳, 🔴, 🔵, ⚫
# edgeshapes="—", # ⇵, ⇶, ⇄, ⇅, ⇆, ⇇, ⇈, ⇉, ⇊, ⬱, —, –, ⟵, ⟶, ➖, −, ➡, ⬅, ⬆, ⬇

#############################################################################
# edge labels
#

default_edge_textsize(b::Backend) = 30

function default_edge_labels(b::Backend, g::AbstractGraph)
  return fill("", ne(g))
end

function default_edge_labels(b::Backend, g::AbstractMetaGraph)
  return IndexLabels(b)
end

default_dims(b::Backend) = true
default_tags(b::Backend) = false
default_ids(b::Backend) = false
default_plevs(b::Backend) = true
default_qns(b::Backend) = false
default_newlines(b::Backend) = true

abstract type AbstractEdgeLabels end

(l::AbstractEdgeLabels)(g::AbstractGraph) = edge_labels(l, g)

struct IndexLabels <: AbstractEdgeLabels
  dims::Bool
  tags::Bool
  ids::Bool
  plevs::Bool
  qns::Bool
  newlines::Bool
end

IndexLabels(; kwargs...) = IndexLabels(Backend(); kwargs...)
IndexLabels(backend; kwargs...) = IndexLabels(Backend(backend); kwargs...)

function IndexLabels(
  b::Backend;
  dims=default_dims(b),
  tags=default_tags(b),
  ids=default_ids(b),
  plevs=default_plevs(b),
  qns=default_qns(b),
  newlines=default_newlines(b),
)
  return IndexLabels(dims, tags, ids, plevs, qns, newlines)
end

edge_labels(b::Backend, l::Vector{String}, g::AbstractGraph) = l

function edge_labels(b::Backend, l::IndexLabels, g::AbstractGraph)
  return edge_labels(l, g)
end

function edge_labels(l::IndexLabels, g::AbstractGraph)
  return String[edge_label(l, g, e) for e in edges(g)]
end

function edge_labels(b::Backend, params::NamedTuple, g::AbstractGraph)
  return IndexLabels(b; params...)(g)
end

function edge_label(l::IndexLabels, g::AbstractMetaGraph, e)
  indsₑ = get_prop(g, e, :inds)
  return label_string(
    indsₑ;
    is_self_loop=is_self_loop(e),
    dims=l.dims,
    tags=l.tags,
    ids=l.ids,
    plevs=l.plevs,
    qns=l.qns,
    newlines=l.newlines,
  )
end

function _edge_label(l, g::AbstractGraph, e)
  return string(e)
end

edge_label(l::IndexLabels, g::AbstractGraph, e) = _edge_label(l, g, e)
edge_label(l, g::AbstractGraph, e) = _edge_label(l, g, e)

#function default_edge_labels(b::Backend, g; kwargs...)
#  return [edge_label(g, e; kwargs...) for e in edges(g)]
#end

plevstring(i::Index) = ITensors.primestring(plev(i))
idstring(i::Index) = string(id(i) % 1000)
tagsstring(i::Index) = string(tags(i))
qnstring(i::Index) = ""
function qnstring(i::QNIndex)
  str = "["
  for (n, qnblock) in pairs(space(i))
    str *= "$qnblock"
    if n ≠ lastindex(space(i))
      str *= ", "
    end
  end
  str *= "]"
  if dir(i) == ITensors.In
    str *= "†"
  end
  return str
end

function label_string(i::Index; dims, tags, plevs, ids, qns)
  showing_plev = plevs && (plev(i) > 0)

  str = ""
  if any((tags, showing_plev, ids, qns))
    str *= "("
  end
  if dims
    str *= string(dim(i))
  end
  if ids
    if dims
      str *= "|"
    end
    str *= idstring(i)
  end
  if tags
    if any((dims, ids))
      str *= "|"
    end
    str *= tagsstring(i)
  end
  if any((tags, showing_plev, ids, qns))
    str *= ")"
  end
  if plevs
    str *= plevstring(i)
  end
  if qns
    str *= qnstring(i)
  end
  return str
end

function label_string(is; is_self_loop=false, dims, tags, plevs, ids, qns, newlines)
  str = ""
  for n in eachindex(is)
    str *= label_string(is[n]; dims=dims, tags=tags, plevs=plevs, ids=ids, qns=qns)
    if n ≠ lastindex(is)
      if any((dims, tags, ids, qns))
        str *= "⊗"
      end
      if newlines && any((tags, ids, qns))
        str *= "\n"
      end
    end
  end
  return str
end

#############################################################################
# edge width
#

function width(inds)
  return log2(dim(inds)) + 1
end

function default_edge_widths(b::Backend, g::AbstractMetaGraph)
  return Float64[width(get_prop(g, e, :inds)) for e in edges(g)]
end

function default_edge_widths(b::Backend, g::AbstractGraph)
  return fill(one(Float64), ne(g))
end

#############################################################################
# arrow
#

default_arrow_size(b::Backend, g) = 30

_hasqns(tn::Vector{ITensor}) = any(hasqns, tn)

function _hasqns(g::AbstractMetaGraph)
  if iszero(ne(g))
    if has_prop(g, first(vertices(g)), :inds)
      return hasqns(get_prop(g, first(vertices(g)), :inds))
    else
      return hasqns(())
    end
  end
  return hasqns(get_prop(g, first(edges(g)), :inds))
end

_hasqns(g::AbstractGraph) = false

default_arrow_show(b::Backend, g) = _hasqns(g)

#############################################################################
# self-loop/siteinds direction
#

default_siteinds_direction(b::Backend, g) = Point2(0, -1)

#############################################################################
# dimensions
#

_ndims(::Any) = 2
_ndims(::NetworkLayout.AbstractLayout{N}) where {N} = N
