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

default_vertex_labels_prefix(b::Backend, g) = "T"
function default_vertex_labels(b::Backend, g::AbstractGraph, vertex_labels_prefix=default_vertex_labels_prefix(b))
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
  return IndexLabels(b)
end

default_dims(b::Backend) = true
default_tags(b::Backend) = false
default_ids(b::Backend) = false
default_plevs(b::Backend) = false
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

function IndexLabels(b::Backend;
  dims=default_dims(b),
  tags=default_tags(b),
  ids=default_ids(b),
  plevs=default_plevs(b),
  qns=default_qns(b),
  newlines=default_newlines(b)
)
  return IndexLabels(dims, tags, ids, plevs, qns, newlines)
end

function edge_labels(b::Backend, l::IndexLabels, g::AbstractGraph)
  return edge_labels(l, g)
end

function edge_labels(l::IndexLabels, g::AbstractGraph)
  return [edge_label(l, g, e) for e in edges(g)]
end

function edge_labels(b::Backend, params::NamedTuple, g::AbstractGraph)
  return IndexLabels(b; params...)(g)
end

function edge_label(l::IndexLabels, g, e)
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
  str = ""
  if any((tags, plevs, ids, qns))
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
  if any((tags, plevs, ids, qns))
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
    str *= label_string(is[n]; dims, tags, plevs, ids, qns)
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

function default_edge_widths(b::Backend, g::AbstractGraph)
  return [width(get_prop(g, e, :inds)) for e in edges(g)]
end

#############################################################################
# arrow
#

default_arrow_size(b::Backend, g) = 30

_hasqns(tn::Vector{ITensor}) = any(hasqns, tn)
function _hasqns(g::AbstractGraph)
  if iszero(ne(g))
    if has_prop(g, first(vertices(g)), :inds)
      return hasqns(get_prop(g, first(vertices(g)), :inds))
    else
      return hasqns(())
    end
  end
  return hasqns(get_prop(g, first(edges(g)), :inds))
end

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
