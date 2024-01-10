
struct CategoryProduct{Labels} <: AbstractCategory
  labels::Labels
  global _CategoryProduct(l) = new{typeof(l)}(l)
end

CategoryProduct(c::CategoryProduct) = _CategoryProduct(labels(c))

labels(s::CategoryProduct) = s.labels

Base.isempty(S::CategoryProduct) = isempty(labels(S))
Base.length(S::CategoryProduct) = length(labels(S))
Base.getindex(S::CategoryProduct, args...) = getindex(labels(S), args...)

function fusion_rule(s1::CategoryProduct, s2::CategoryProduct)
  return [CategoryProduct(l) for l in labels_fusion_rule(labels(s1), labels(s2))]
end

Base.:(==)(A::CategoryProduct, B::CategoryProduct) = labels_equal(labels(A), labels(B))

const Sector = CategoryProduct

function Base.show(io::IO, s::CategoryProduct)
  (length(s) < 2) && print(io, "Sector")
  print(io, "(")
  symbol = ""
  for p in pairs(labels(s))
    print(io, symbol)
    label_show(io, p[1], p[2])
    symbol = " × "
  end
  return print(io, ")")
end

label_show(io::IO, k, v) = print(io, v)

label_show(io::IO, k::Symbol, v) = print(io, "($k=$v,)")

×(c1::AbstractCategory, c2::AbstractCategory) = ×(CategoryProduct(c1), CategoryProduct(c2))
function ×(p1::CategoryProduct, p2::CategoryProduct)
  return CategoryProduct(labels_product(labels(p1), labels(p2)))
end

labels_product(l1::NamedTuple, l2::NamedTuple) = nt_union(l1, l2)

labels_product(l1::Tuple, l2::Tuple) = (l1..., l2...)

×(nt1::NamedTuple, nt2::NamedTuple) = ×(CategoryProduct(nt1), CategoryProduct(nt2))
×(c1::NamedTuple, c2::AbstractCategory) = ×(CategoryProduct(c1), CategoryProduct(c2))
×(c1::AbstractCategory, c2::NamedTuple) = ×(CategoryProduct(c1), CategoryProduct(c2))

#
# Dictionary-like implementation
#

function CategoryProduct(nt::NamedTuple)
  labels = nt_sort(nt)
  return _CategoryProduct(labels)
end

CategoryProduct(; kws...) = CategoryProduct((; kws...))

function CategoryProduct(pairs::Pair...)
  keys = ntuple(n -> Symbol(pairs[n][1]), length(pairs))
  vals = ntuple(n -> pairs[n][2], length(pairs))
  return CategoryProduct(NamedTuple{keys}(vals))
end

function labels_fusion_rule(A::NamedTuple, B::NamedTuple)
  qs = [A]
  for (la, lb) in zip(pairs(nt_intersect(A, B)), pairs(nt_intersect(B, A)))
    @assert la[1] == lb[1]
    fused_vals = ⊗(la[2], lb[2])
    qs = [nt_union((; la[1] => v), q) for v in fused_vals for q in qs]
  end
  # Include sectors of B not in A
  qs = [nt_union(q, B) for q in qs]
  return qs
end

function labels_equal(A::NamedTuple, B::NamedTuple)
  common_labels = zip(pairs(nt_intersect(A, B)), pairs(nt_intersect(B, A)))
  common_labels_match = all(nl -> (nl[1] == nl[2]), common_labels)
  unique_labels_zero = all(l -> istrivial(l), nt_symdiff(A, B))
  return common_labels_match && unique_labels_zero
end

#
# Ordered implementation
#

CategoryProduct(t::Tuple) = _CategoryProduct(t)
CategoryProduct(cats::AbstractCategory...) = CategoryProduct((cats...,))

labels_equal(o1::Tuple, o2::Tuple) = (o1 == o2)

function labels_fusion_rule(o1::Tuple, o2::Tuple)
  N = length(o1)
  length(o2) == N ||
    throw(DimensionMismatch("Ordered CategoryProduct must have same size in ⊗"))
  os = [o1]
  replace(o, n, val) = ntuple(m -> (m == n) ? val : o[m], length(o))
  for n in 1:N
    os = [replace(o, n, f) for f in ⊗(o1[n], o2[n]) for o in os]
  end
  return os
end
