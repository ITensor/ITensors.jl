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

trivial(type::Type{<:CategoryProduct}) = sector(categories_trivial(categories_type(type)))

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
  return categories_isless(categories(s1), categories(s2))
end
function Base.isless(s1::CategoryProduct, s2::CategoryProduct)
  return categories_isless(categories(s1), categories(s2))
end

# =======================================  shared  =========================================
# there are 2 implementations for CategoryProduct
# - ordered-like with a Tuple
# - dictionary-like with a NamedTuple

function categories_fusion_rule(cats1, cats2)
  diff_cat = CategoryProduct(find_diff(cats1, cats2))
  shared1, shared2 = find_common(cats1, cats2)
  fused = map(fusion_rule, values(shared1), values(shared2))
  return recover_key(typeof(shared1), fused) × diff_cat
end

# get clean results when mixing implementations
categories_isequal(::Tuple, ::NamedTuple) = false
categories_isequal(::NamedTuple, ::Tuple) = false

categories_isless(::NamedTuple, ::Tuple) = throw(ArgumentError("Not implemented"))
categories_isless(::Tuple, ::NamedTuple) = throw(ArgumentError("Not implemented"))

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

categories_isless(::Tuple, ::Tuple) = throw(ArgumentError("Not implemented"))
categories_isless(t1::T, t2::T) where {T<:Tuple} = t1 < t2

categories_product(l1::Tuple, l2::Tuple) = (l1..., l2...)

categories_trivial(type::Type{<:Tuple}) = trivial.(fieldtypes(type))

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

CategoryProduct(; kws...) = CategoryProduct((; kws...))

# avoid having 2 different kinds of EmptyCategory: cast empty NamedTuple to Tuple{}
CategoryProduct(::NamedTuple{()}) = CategoryProduct(())

function CategoryProduct(pairs::Pair...)
  keys = ntuple(n -> Symbol(pairs[n][1]), length(pairs))
  vals = ntuple(n -> pairs[n][2], length(pairs))
  return CategoryProduct(NamedTuple{keys}(vals))
end

# sector() acts as empty NamedTuple
function categories_isequal(nt::NamedTuple, ::Tuple{})
  return categories_isequal(nt, categories_trivial(typeof(nt)))
end
function categories_isequal(::Tuple{}, nt::NamedTuple)
  return categories_isequal(categories_trivial(typeof(nt)), nt)
end
function categories_isequal(nt1::NamedTuple, nt2::NamedTuple)
  return ==(sym_categories_insert_unspecified(nt1, nt2)...)
end

function categories_isless(nt::NamedTuple, ::Tuple{})
  return categories_isless(nt, categories_trivial(typeof(nt)))
end
function categories_isless(::Tuple{}, nt::NamedTuple)
  return categories_isless(categories_trivial(typeof(nt)), nt)
end
function categories_isless(nt1::NamedTuple, nt2::NamedTuple)
  return isless(sym_categories_insert_unspecified(nt1, nt2)...)
end

function sym_categories_insert_unspecified(nt1::NamedTuple, nt2::NamedTuple)
  return categories_insert_unspecified(nt1, nt2), categories_insert_unspecified(nt2, nt1)
end

function categories_insert_unspecified(nt1::NamedTuple, nt2::NamedTuple)
  diff1 = categories_trivial(typeof(setdiff_keys(nt2, nt1)))
  return sort_keys(union_keys(nt1, diff1))
end

categories_product(l1::NamedTuple, ::Tuple{}) = l1
categories_product(::Tuple{}, l2::NamedTuple) = l2
function categories_product(l1::NamedTuple, l2::NamedTuple)
  if length(intersect_keys(l1, l2)) > 0
    throw(ArgumentError("Cannot define product of shared keys"))
  end
  return union_keys(l1, l2)
end

function categories_trivial(type::Type{<:NamedTuple{Keys}}) where {Keys}
  return NamedTuple{Keys}(trivial.(fieldtypes(type)))
end

function find_common(nt1::NamedTuple, nt2::NamedTuple)
  return intersect_keys(nt1, nt2), intersect_keys(nt2, nt1)
end

find_diff(nt1::NamedTuple, nt2::NamedTuple) = symdiff_keys(nt1, nt2)
