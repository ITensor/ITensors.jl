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

function SymmetryStyle(nt::NamedTuple)
  return if length(nt) == 0
    EmptyCategory()
  else
    reduce(combine_styles, map(SymmetryStyle, (values(nt))))
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

# edge cases
categories_product(l1::NamedTuple, l2::Tuple{}) = l1
categories_product(l1::Tuple{}, l2::NamedTuple) = l2

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

# allow ⊗ for different types in NamedTuple
function fusion_rule(
  s1::CategoryProduct{Cat1}, s2::CategoryProduct{Cat2}
) where {Cat1<:NamedTuple,Cat2<:NamedTuple}

  # avoid issues with length 0 CategoryProduct
  if SymmetryStyle(s1) == EmptyCategory()
    if SymmetryStyle(s2) == AbelianGroup() || SymmetryStyle(s2) == EmptyCategory()
      return s2
    end
    return GradedAxes.gradedrange([s2 => 1])
  end
  if SymmetryStyle(s2) == EmptyCategory()
    if SymmetryStyle(s1) == AbelianGroup()
      return s1
    end
    return GradedAxes.gradedrange([s1 => 1])
  end

  cats1 = categories(s1)
  cats2 = categories(s2)
  diff_cat = CategoryProduct(symdiff_keys(cats1, cats2))
  shared1 = intersect_keys(cats1, cats2)
  if length(shared1) == 0
    if SymmetryStyle(diff_cat) == AbelianGroup()
      return diff_cat
    end
    return GradedAxes.gradedrange([diff_cat => 1])
  end

  shared2 = intersect_keys(cats2, cats1)
  fused = fusion_rule(shared1, shared2)
  out = diff_cat × fused
  return out
end

function fusion_rule(
  cats1::NT, cats2::NT
) where {Names,NT<:NamedTuple{Names,<:Tuple{AbstractCategory,Vararg{AbstractCategory}}}}
  return fusion_rule(cats1[(Names[1],)], cats2[(Names[1],)]) ×
         fusion_rule(cats1[Names[2:end]], cats2[Names[2:end]])
end

fusion_rule(cats1::NamedTuple{}, cats2::NamedTuple{}) = sector()

function fusion_rule(
  cats1::NT, cats2::NT
) where {NT<:NamedTuple{<:Any,<:Tuple{AbstractCategory}}}
  # cannot be EmptyCategory
  key = only(keys(cats1))
  fused = only(values(cats1)) ⊗ only(values(cats2))
  if SymmetryStyle(cats1) == AbelianGroup()
    return sector(key => fused)
  end
  la = fused[1]
  v = Vector{Pair{CategoryProduct{NT},Int64}}()
  for la in blocklengths(fused)
    push!(v, sector(key => LabelledNumbers.label(la)) => LabelledNumbers.unlabel(la))
  end
  g = GradedAxes.gradedrange(v)
  return g
end

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
  cats1 = categories(s1)
  cats2 = categories(s2)
  fused = map(fusion_rule, cats1, cats2)
  g = reduce(×, fused)
  return g
end
