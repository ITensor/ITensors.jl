# This files defines a structure for Cartesian product of 2 or more fusion categories
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

# ==============  Definition and getters  =================
struct CategoryProduct{Categories} <: AbstractCategory
  cats::Categories
  global _CategoryProduct(l) = new{typeof(l)}(l)
end

CategoryProduct(c::CategoryProduct) = _CategoryProduct(categories(c))

categories(s::CategoryProduct) = s.cats

# ==============  SymmetryStyle ==============================
combine_styles(::AbelianGroup, ::AbelianGroup) = AbelianGroup()
combine_styles(::AbelianGroup, ::NonAbelianGroup) = NonAbelianGroup()
combine_styles(::AbelianGroup, ::NonGroupCategory) = NonGroupCategory()
combine_styles(::NonAbelianGroup, ::AbelianGroup) = NonAbelianGroup()
combine_styles(::NonAbelianGroup, ::NonAbelianGroup) = NonAbelianGroup()
combine_styles(::NonAbelianGroup, ::NonGroupCategory) = NonGroupCategory()
combine_styles(::NonGroupCategory, ::SymmetryStyle) = NonGroupCategory()

function SymmetryStyle(c::CategoryProduct)
  return if length(categories(c)) == 0
    EmptyCategory()
  else
    reduce(combine_styles, map(SymmetryStyle, (categories(c))))
  end
end

# ==============  Sector interface  =================
function quantum_dimension(::NonAbelianGroup, s::CategoryProduct)
  return prod(map(quantum_dimension, categories(s)))
end

function quantum_dimension(::NonGroupCategory, s::CategoryProduct)
  return prod(map(quantum_dimension, categories(s)))
end

GradedAxes.dual(s::CategoryProduct) = CategoryProduct(map(GradedAxes.dual, categories(s)))

# ==============  Base interface  =================
function Base.:(==)(A::CategoryProduct, B::CategoryProduct)
  return categories_equal(categories(A), categories(B))
end

function Base.show(io::IO, s::CategoryProduct)
  (length(categories(s)) < 2) && print(io, "sector")
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

# ==============  Cartesian product  =================
×(c1::AbstractCategory, c2::AbstractCategory) = ×(CategoryProduct(c1), CategoryProduct(c2))
function ×(p1::CategoryProduct, p2::CategoryProduct)
  return CategoryProduct(categories_product(categories(p1), categories(p2)))
end

# currently (A=U1(1),) × (A=U1(2),) = sector((A=U1(1),))
# this is misleading. TBD throw in this case?
categories_product(l1::NamedTuple, l2::NamedTuple) = union_keys(l1, l2)

categories_product(l1::Tuple, l2::Tuple) = (l1..., l2...)

×(nt1::NamedTuple, nt2::NamedTuple) = ×(CategoryProduct(nt1), CategoryProduct(nt2))
×(c1::NamedTuple, c2::AbstractCategory) = ×(CategoryProduct(c1), CategoryProduct(c2))
×(c1::AbstractCategory, c2::NamedTuple) = ×(CategoryProduct(c1), CategoryProduct(c2))

function ×(l1::LabelledNumbers.LabelledInteger, l2::LabelledNumbers.LabelledInteger)
  c3 = LabelledNumbers.label(l1) × LabelledNumbers.label(l2)
  m3 = LabelledNumbers.unlabel(l1) * LabelledNumbers.unlabel(l2)
  return LabelledNumbers.LabelledInteger(m3, c3)
end

×(g::AbstractUnitRange, c::AbstractCategory) = ×(g, GradedAxes.gradedrange([c => 1]))
×(c::AbstractCategory, g::AbstractUnitRange) = ×(GradedAxes.gradedrange([c => 1]), g)

function ×(g1::GradedAxes.GradedUnitRange, g2::GradedAxes.GradedUnitRange)
  # keep F convention in loop order
  v = [
    l1 × l2 for l2 in BlockArrays.blocklengths(g2) for l1 in BlockArrays.blocklengths(g1)
  ]
  return GradedAxes.gradedrange(v)
end

# ==============  Dictionary-like implementation  =================
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

function categories_equal(A::NamedTuple, B::NamedTuple)
  common_categories = zip(pairs(intersect_keys(A, B)), pairs(intersect_keys(B, A)))
  common_categories_match = all(nl -> (nl[1] == nl[2]), common_categories)
  unique_categories_zero = all(l -> istrivial(l), symdiff_keys(A, B))
  return common_categories_match && unique_categories_zero
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

# allow ⊗ for different types in NamedTuple
function fusion_rule(
  s1::CategoryProduct{Cat1}, s2::CategoryProduct{Cat2}
) where {Cat1<:NamedTuple,Cat2<:NamedTuple} end

# ==============  Ordered implementation  =================
CategoryProduct(t::Tuple) = _CategoryProduct(t)
CategoryProduct(cats::AbstractCategory...) = CategoryProduct((cats...,))

categories_equal(o1::Tuple, o2::Tuple) = (o1 == o2)

sector(args...; kws...) = CategoryProduct(args...; kws...)

# for ordered tuple, impose same type in fusion
function fusion_rule(s1::CategoryProduct{Cat}, s2::CategoryProduct{Cat}) where {Cat<:Tuple}
  if SymmetryStyle(s1) == EmptyCategory()  # compile-time; simpler than specifying init
    return s1
  end
  cat1 = categories(s1)
  cat2 = categories(s2)
  prod12 = ntuple(i -> cat1[i] ⊗ cat2[i], length(cat1))
  g = reduce(×, prod12)
  return g
end
