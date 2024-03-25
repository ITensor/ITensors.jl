# This files defines a structure for Cartesian product of 2 or more fusion categories
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

struct CategoryProduct{Categories} <: AbstractCategory
  cats::Categories
  global _CategoryProduct(l) = new{typeof(l)}(l)
end

CategoryProduct(c::CategoryProduct) = _CategoryProduct(categories(c))

categories(s::CategoryProduct) = s.cats

Base.isempty(s::CategoryProduct) = isempty(categories(s))
Base.length(s::CategoryProduct) = length(categories(s))
Base.getindex(s::CategoryProduct, args...) = getindex(categories(s), args...)

function dimension(s::CategoryProduct)
  if length(s) == 0
    return 0
  end
  return prod(map(dimension, categories(s)))
end

GradedAxes.dual(s::CategoryProduct) = CategoryProduct(map(GradedAxes.dual, categories(s)))

function fusion_rule(s1::CategoryProduct, s2::CategoryProduct)
  return [
    CategoryProduct(l) for l in categories_fusion_rule(categories(s1), categories(s2))
  ]
end

function Base.:(==)(A::CategoryProduct, B::CategoryProduct)
  return categories_equal(categories(A), categories(B))
end

function Base.show(io::IO, s::CategoryProduct)
  (length(s) < 2) && print(io, "sector")
  print(io, "(")
  symbol = ""
  for p in pairs(categories(s))
    print(io, symbol)
    category_show(io, p[1], p[2])
    symbol = " × "
  end
  return print(io, ")")
end

category_show(io::IO, k, v) = print(io, v)

category_show(io::IO, k::Symbol, v) = print(io, "($k=$v,)")

×(c1::AbstractCategory, c2::AbstractCategory) = ×(CategoryProduct(c1), CategoryProduct(c2))
function ×(p1::CategoryProduct, p2::CategoryProduct)
  return CategoryProduct(categories_product(categories(p1), categories(p2)))
end

categories_product(l1::NamedTuple, l2::NamedTuple) = union_keys(l1, l2)

categories_product(l1::Tuple, l2::Tuple) = (l1..., l2...)

×(nt1::NamedTuple, nt2::NamedTuple) = ×(CategoryProduct(nt1), CategoryProduct(nt2))
×(c1::NamedTuple, c2::AbstractCategory) = ×(CategoryProduct(c1), CategoryProduct(c2))
×(c1::AbstractCategory, c2::NamedTuple) = ×(CategoryProduct(c1), CategoryProduct(c2))

#
# Dictionary-like implementation
#

function CategoryProduct(nt::NamedTuple)
  categories = sort_keys(nt)
  return _CategoryProduct(categories)
end

CategoryProduct(; kws...) = CategoryProduct((; kws...))

function CategoryProduct(pairs::Pair...)
  keys = ntuple(n -> Symbol(pairs[n][1]), length(pairs))
  vals = ntuple(n -> pairs[n][2], length(pairs))
  return CategoryProduct(NamedTuple{keys}(vals))
end

function categories_fusion_rule(A::NamedTuple, B::NamedTuple)
  qs = [A]
  for (la, lb) in zip(pairs(intersect_keys(A, B)), pairs(intersect_keys(B, A)))
    @assert la[1] == lb[1]
    fused_vals = ⊗(la[2], lb[2])
    qs = [union_keys((; la[1] => v), q) for v in fused_vals for q in qs]
  end
  # Include sectors of B not in A
  qs = [union_keys(q, B) for q in qs]
  return qs
end

function categories_equal(A::NamedTuple, B::NamedTuple)
  common_categories = zip(pairs(intersect_keys(A, B)), pairs(intersect_keys(B, A)))
  common_categories_match = all(nl -> (nl[1] == nl[2]), common_categories)
  unique_categories_zero = all(l -> istrivial(l), symdiff_keys(A, B))
  return common_categories_match && unique_categories_zero
end

#
# Ordered implementation
#

CategoryProduct(t::Tuple) = _CategoryProduct(t)
CategoryProduct(cats::AbstractCategory...) = CategoryProduct((cats...,))

categories_equal(o1::Tuple, o2::Tuple) = (o1 == o2)

function categories_fusion_rule(o1::Tuple, o2::Tuple)
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

sector(args...; kws...) = CategoryProduct(args...; kws...)
