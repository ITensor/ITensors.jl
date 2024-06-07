# This files defines a structure for Cartesian product of 2 or more fusion categories
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

# =====================================  Definition  =======================================
struct CategoryProduct{Categories} <: AbstractCategory
  cats::Categories
  global _CategoryProduct(l) = new{typeof(l)}(l)
end

CategoryProduct(c::CategoryProduct) = _CategoryProduct(categories(c))

categories(s::CategoryProduct) = s.cats

# =================================  Sectors interface  ====================================
function SymmetryStyle(c::CategoryProduct)
  return reduce(combine_styles, map(SymmetryStyle, categories(c)); init=EmptyCategory())
end

function quantum_dimension(::NonAbelianGroup, s::CategoryProduct)
  return prod(map(quantum_dimension, categories(s)))
end

function quantum_dimension(::NonGroupCategory, s::CategoryProduct)
  return prod(map(quantum_dimension, categories(s)))
end

GradedAxes.dual(s::CategoryProduct) = CategoryProduct(map(GradedAxes.dual, categories(s)))

function trivial(type::Type{<:CategoryProduct})
  cat_type = categories_type(type)
  return recover_key(cat_type, categories_trivial(cat_type))
end

# ===================================  Base interface  =====================================
function Base.:(==)(A::CategoryProduct, B::CategoryProduct)
  return categories_isequal(categories(A), categories(B))
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

category_show(io::IO, ::Int, v) = print(io, v)
category_show(io::IO, k::Symbol, v) = print(io, "($k=$v,)")

function Base.isless(s1::C, s2::C) where {C<:CategoryProduct}
  return isless(
    category_label.(values(categories(s1))), category_label.(values(categories(s2)))
  )
end

# =======================================  shared  =========================================
# there are 2 implementations for CategoryProduct
# - ordered-like with a Tuple
# - dictionary-like with a NamedTuple

function categories_fusion_rule(cats1, cats2)
  diff_cat = CategoryProduct(find_diff(cats1, cats2))
  nt1, nt2 = find_common(cats1, cats2)
  fused = map(fusion_rule, values(nt1), values(nt2))
  return recover_key(typeof(nt1), fused) × diff_cat
end

categories_isequal(::Tuple, ::NamedTuple) = false
categories_isequal(::NamedTuple, ::Tuple) = false

categories_trivial(cat_type::Type) = trivial.(fieldtypes(cat_type))

categories_type(::Type{<:CategoryProduct{T}}) where {T} = T

recover_key(T::Type, t::Tuple{Vararg{<:AbstractCategory}}) = sector(T, t)
recover_key(T::Type, c::AbstractCategory) = recover_key(T, (c,))
recover_key(T::Type, c::CategoryProduct) = recover_key(T, categories(c))

function recover_key(T::Type, fused::Tuple)
  # here fused contains at leat one GradedUnitRange
  g0 = reduce(×, fused)
  # convention: keep unsorted blocklabels as produced by F order loops in ×
  new_labels = recover_key.(T, GradedAxes.blocklabels(g0))
  new_blocklengths =
    LabelledNumbers.labelled.(GradedAxes.unlabel.(BlockArrays.blocklengths(g0)), new_labels)
  return GradedAxes.gradedrange(new_blocklengths)
end

sector(T::Type{<:CategoryProduct}, cats::Tuple) = sector(categories_type(T), cats)
sector(T::Type, cats::Tuple) = sector(T(cats))  # recover NamedTuple
sector(args...; kws...) = CategoryProduct(args...; kws...)

# =================================  Cartesian Product  ====================================
×(c1::AbstractCategory, c2::AbstractCategory) = ×(CategoryProduct(c1), CategoryProduct(c2))
function ×(p1::CategoryProduct, p2::CategoryProduct)
  return CategoryProduct(categories_product(categories(p1), categories(p2)))
end

function categories_product(l1::NamedTuple, l2::NamedTuple)
  if length(intersect_keys(l1, l2)) > 0
    throw(error("Cannot define product of shared keys"))
  end
  return union_keys(l1, l2)
end
categories_product(l1::Tuple, l2::Tuple) = (l1..., l2...)

# edge cases
categories_product(l1::NamedTuple, ::Tuple{}) = l1
categories_product(::Tuple{}, l2::NamedTuple) = l2

×(a, g::AbstractUnitRange) = ×(to_graded_axis(a), g)
×(g::AbstractUnitRange, b) = ×(g, to_graded_axis(b))
×(nt1::NamedTuple, nt2::NamedTuple) = ×(CategoryProduct(nt1), CategoryProduct(nt2))
×(c1::NamedTuple, c2::AbstractCategory) = ×(CategoryProduct(c1), CategoryProduct(c2))
×(c1::AbstractCategory, c2::NamedTuple) = ×(CategoryProduct(c1), CategoryProduct(c2))

function ×(l1::LabelledNumbers.LabelledInteger, l2::LabelledNumbers.LabelledInteger)
  c3 = LabelledNumbers.label(l1) × LabelledNumbers.label(l2)
  m3 = LabelledNumbers.unlabel(l1) * LabelledNumbers.unlabel(l2)
  return LabelledNumbers.labelled(m3, c3)
end

function ×(g1::AbstractUnitRange, g2::AbstractUnitRange)
  v = map(
    ((l1, l2),) -> l1 × l2,
    Iterators.flatten((
      Iterators.product(BlockArrays.blocklengths(g1), BlockArrays.blocklengths(g2)),
    ),),
  )
  return GradedAxes.gradedrange(v)
end

# ====================================  Fusion rules  ======================================
# generic case: fusion returns a GradedAxes, even for fusion with Empty
function fusion_rule(::SymmetryStyle, s1::CategoryProduct, s2::CategoryProduct)
  return to_graded_axis(categories_fusion_rule(categories(s1), categories(s2)))
end

# Abelian case: fusion returns CategoryProduct
function fusion_rule(::AbelianGroup, s1::CategoryProduct, s2::CategoryProduct)
  return categories_fusion_rule(categories(s1), categories(s2))
end

# Empty case
function fusion_rule(
  ::EmptyCategory, ::CategoryProduct{Tuple{}}, ::CategoryProduct{Tuple{}}
)
  return sector()
end

# EmptyCategory acts as trivial on any AbstractCategory, not just CategoryProduct
function fusion_rule(::SymmetryStyle, ::CategoryProduct{Tuple{}}, c::AbstractCategory)
  return to_graded_axis(c)
end
function fusion_rule(::SymmetryStyle, ::CategoryProduct{Tuple{}}, c::CategoryProduct)
  return to_graded_axis(c)
end
function fusion_rule(::SymmetryStyle, c::AbstractCategory, ::CategoryProduct{Tuple{}})
  return to_graded_axis(c)
end
function fusion_rule(::SymmetryStyle, c::CategoryProduct, ::CategoryProduct{Tuple{}})
  return to_graded_axis(c)
end

# abelian case: return Category
fusion_rule(::AbelianGroup, ::CategoryProduct{Tuple{}}, c::AbstractCategory) = c
fusion_rule(::AbelianGroup, ::CategoryProduct{Tuple{}}, c::CategoryProduct) = c
fusion_rule(::AbelianGroup, c::AbstractCategory, ::CategoryProduct{Tuple{}}) = c
fusion_rule(::AbelianGroup, c::CategoryProduct, ::CategoryProduct{Tuple{}}) = c

# ===============================  Ordered implementation  =================================
CategoryProduct(t::Tuple) = _CategoryProduct(t)
CategoryProduct(cats::AbstractCategory...) = CategoryProduct((cats...,))

categories_isequal(o1::Tuple, o2::Tuple) = (o1 == o2)

function find_common(t1::Tuple, t2::Tuple)
  n = min(length(t1), length(t2))
  return t1[begin:n], t2[begin:n]
end

function find_diff(t1::Tuple, t2::Tuple)
  n1 = length(t1)
  n2 = length(t2)
  return n1 < n2 ? t2[(n1 + 1):end] : t1[(n2 + 1):end]
end

# ===========================  Dictionary-like implementation  =============================
function CategoryProduct(nt::NamedTuple)
  categories = sort_keys(nt)
  return _CategoryProduct(categories)
end

# avoid having 2 different kinds of EmptyCategory: cast empty NamedTuple to Tuple{}
CategoryProduct(::NamedTuple{()}) = CategoryProduct(())

CategoryProduct(; kws...) = CategoryProduct((; kws...))

function CategoryProduct(pairs::Pair...)
  keys = ntuple(n -> Symbol(pairs[n][1]), length(pairs))
  vals = ntuple(n -> pairs[n][2], length(pairs))
  return CategoryProduct(NamedTuple{keys}(vals))
end

function categories_isequal(A::NamedTuple, B::NamedTuple)
  sharedA, sharedB = find_common(A, B)
  common_categories_match = sharedA == sharedB
  unique_categories_zero = all(map(istrivial, find_diff(A, B)))
  return common_categories_match && unique_categories_zero
end

function find_common(nt1::NamedTuple, nt2::NamedTuple)
  return intersect_keys(nt1, nt2), intersect_keys(nt2, nt1)
end

find_diff(nt1::NamedTuple, nt2::NamedTuple) = symdiff_keys(nt1, nt2)
