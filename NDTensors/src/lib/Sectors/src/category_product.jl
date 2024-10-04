# This files defines a structure for Cartesian product of 2 or more fusion categories
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

using BlockArrays: blocklengths
using ..LabelledNumbers: LabelledInteger, label, labelled, unlabel
using ..GradedAxes: AbstractGradedUnitRange, GradedAxes, dual

# =====================================  Definition  =======================================
struct CategoryProduct{Categories} <: AbstractCategory
  cats::Categories
  global _CategoryProduct(l) = new{typeof(l)}(l)
end

CategoryProduct(c::CategoryProduct) = _CategoryProduct(categories(c))

categories(s::CategoryProduct) = s.cats

const TrivialSector{Categories<:Union{Tuple{},NamedTuple{()}}} = CategoryProduct{Categories}
TrivialSector() = CategoryProduct(())

# =================================  Sectors interface  ====================================
SymmetryStyle(T::Type{<:CategoryProduct}) = categories_symmetrystyle(categories_type(T))

function quantum_dimension(::NotAbelianStyle, s::CategoryProduct)
  return mapreduce(quantum_dimension, *, categories(s))
end

# use map instead of broadcast to support both Tuple and NamedTuple
GradedAxes.dual(s::CategoryProduct) = CategoryProduct(map(dual, categories(s)))

function trivial(type::Type{<:CategoryProduct})
  return CategoryProduct(categories_trivial(categories_type(type)))
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

function Base.isless(s1::CategoryProduct, s2::CategoryProduct)
  return categories_isless(categories(s1), categories(s2))
end

# =======================================  shared  =========================================
# there are 2 implementations for CategoryProduct
# - ordered-like with a Tuple
# - dictionary-like with a NamedTuple

categories_isequal(o1::Tuple, o2::Tuple) = (o1 == o2)
function categories_isequal(nt::NamedTuple, ::Tuple{})
  return categories_isequal(nt, (;))
end
function categories_isequal(::Tuple{}, nt::NamedTuple)
  return categories_isequal((;), nt)
end
function categories_isequal(nt1::NamedTuple, nt2::NamedTuple)
  return ==(sym_categories_insert_unspecified(nt1, nt2)...)
end

# get clean results when mixing implementations
categories_isequal(::Tuple, ::NamedTuple) = false
categories_isequal(::NamedTuple, ::Tuple) = false

categories_isless(::Tuple, ::Tuple) = throw(ArgumentError("Not implemented"))
categories_isless(t1::T, t2::T) where {T<:Tuple} = t1 < t2
function categories_isless(nt::NamedTuple, ::Tuple{})
  return categories_isless(nt, (;))
end
function categories_isless(::Tuple{}, nt::NamedTuple)
  return categories_isless((;), nt)
end
function categories_isless(nt1::NamedTuple, nt2::NamedTuple)
  return isless(sym_categories_insert_unspecified(nt1, nt2)...)
end

categories_isless(::NamedTuple, ::Tuple) = throw(ArgumentError("Not implemented"))
categories_isless(::Tuple, ::NamedTuple) = throw(ArgumentError("Not implemented"))

categories_type(::Type{<:CategoryProduct{T}}) where {T} = T

function categories_fusion_rule(cats1, cats2)
  shared_cat = shared_categories_fusion_rule(categories_common(cats1, cats2)...)
  diff_cat = CategoryProduct(categories_diff(cats1, cats2))
  return shared_cat × diff_cat
end

function recover_style(T::Type, fused)
  style = categories_symmetrystyle(T)
  return recover_category_product_type(style, T, fused)
end

function recover_category_product_type(::AbelianStyle, T::Type, fused)
  return recover_category_product_type(T, fused)
end

function recover_category_product_type(::NotAbelianStyle, T::Type, fused)
  # here fused contains at least one graded unit range.
  # convert eg. Tuple{GradedUnitRange{SU2}, GradedUnitRange{SU2}} into GradedUnitRange{SU2×SU2}
  g = reduce(×, fused)
  # convention: keep unsorted blocklabels as produced by F order loops in ×
  type_fixed = recover_category_product_type(T, g)
  return type_fixed
end

function recover_category_product_type(T::Type, g0::AbstractGradedUnitRange)
  new_labels = recover_category_product_type.(T, blocklabels(g0))
  new_blocklengths = labelled.(unlabel.(blocklengths(g0)), new_labels)
  return gradedrange(new_blocklengths)
end

function recover_category_product_type(T::Type, c::AbstractCategory)
  return recover_category_product_type(T, CategoryProduct(c))
end

function recover_category_product_type(T::Type, c::CategoryProduct)
  return recover_category_product_type(T, categories(c))
end

function recover_category_product_type(T::Type{<:CategoryProduct}, cats)
  return recover_category_product_type(categories_type(T), cats)
end

function recover_category_product_type(T::Type, cats)
  return CategoryProduct(T(cats))
end

# =================================  Cartesian Product  ====================================
×(c1::AbstractCategory, c2::AbstractCategory) = ×(CategoryProduct(c1), CategoryProduct(c2))
function ×(p1::CategoryProduct, p2::CategoryProduct)
  return CategoryProduct(categories_product(categories(p1), categories(p2)))
end

×(a, g::AbstractUnitRange) = ×(to_gradedrange(a), g)
×(g::AbstractUnitRange, b) = ×(g, to_gradedrange(b))
×(nt1::NamedTuple, nt2::NamedTuple) = ×(CategoryProduct(nt1), CategoryProduct(nt2))
×(c1::NamedTuple, c2::AbstractCategory) = ×(CategoryProduct(c1), CategoryProduct(c2))
×(c1::AbstractCategory, c2::NamedTuple) = ×(CategoryProduct(c1), CategoryProduct(c2))

function ×(l1::LabelledInteger, l2::LabelledInteger)
  c3 = label(l1) × label(l2)
  m3 = unlabel(l1) * unlabel(l2)
  return labelled(m3, c3)
end

function ×(g1::AbstractUnitRange, g2::AbstractUnitRange)
  v = map(
    ((l1, l2),) -> l1 × l2,
    Iterators.flatten((Iterators.product(blocklengths(g1), blocklengths(g2)),),),
  )
  return gradedrange(v)
end

# ====================================  Fusion rules  ======================================
# generic case: fusion returns a GradedAxes, even for fusion with Empty
function fusion_rule(::NotAbelianStyle, s1::CategoryProduct, s2::CategoryProduct)
  return to_gradedrange(categories_fusion_rule(categories(s1), categories(s2)))
end

# Abelian case: fusion returns CategoryProduct
function fusion_rule(::AbelianStyle, s1::CategoryProduct, s2::CategoryProduct)
  return categories_fusion_rule(categories(s1), categories(s2))
end

# Empty case
function fusion_rule(::AbelianStyle, ::TrivialSector, ::TrivialSector)
  return CategoryProduct(())
end

# TrivialSector acts as trivial on any AbstractCategory, not just CategoryProduct
function fusion_rule(::NotAbelianStyle, ::TrivialSector, c::AbstractCategory)
  return to_gradedrange(c)
end
function fusion_rule(::NotAbelianStyle, c::AbstractCategory, ::TrivialSector)
  return to_gradedrange(c)
end
function fusion_rule(::NotAbelianStyle, ::TrivialSector, c::CategoryProduct)
  return to_gradedrange(c)
end
function fusion_rule(::NotAbelianStyle, c::CategoryProduct, ::TrivialSector)
  return to_gradedrange(c)
end

# abelian case: return Category
fusion_rule(::AbelianStyle, c::AbstractCategory, ::TrivialSector) = c
fusion_rule(::AbelianStyle, ::TrivialSector, c::AbstractCategory) = c
fusion_rule(::AbelianStyle, c::CategoryProduct, ::TrivialSector) = c
fusion_rule(::AbelianStyle, ::TrivialSector, c::CategoryProduct) = c

# ===============================  Ordered implementation  =================================
CategoryProduct(t::Tuple) = _CategoryProduct(t)
CategoryProduct(cats::AbstractCategory...) = CategoryProduct(cats)

function categories_symmetrystyle(T::Type{<:Tuple})
  return mapreduce(SymmetryStyle, combine_styles, fieldtypes(T); init=AbelianStyle())
end

categories_product(::NamedTuple{()}, l1::Tuple) = l1
categories_product(l2::Tuple, ::NamedTuple{()}) = l2
categories_product(l1::Tuple, l2::Tuple) = (l1..., l2...)

categories_trivial(type::Type{<:Tuple}) = trivial.(fieldtypes(type))

function categories_common(t1::Tuple, t2::Tuple)
  n = min(length(t1), length(t2))
  return t1[begin:n], t2[begin:n]
end

function categories_diff(t1::Tuple, t2::Tuple)
  n1 = length(t1)
  n2 = length(t2)
  return n1 < n2 ? t2[(n1 + 1):end] : t1[(n2 + 1):end]
end

function shared_categories_fusion_rule(shared1::T, shared2::T) where {T<:Tuple}
  fused = map(fusion_rule, shared1, shared2)
  return recover_style(T, fused)
end

# ===========================  Dictionary-like implementation  =============================
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

function categories_symmetrystyle(NT::Type{<:NamedTuple})
  return mapreduce(SymmetryStyle, combine_styles, fieldtypes(NT); init=AbelianStyle())
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

function categories_common(nt1::NamedTuple, nt2::NamedTuple)
  # CategoryProduct(nt::NamedTuple) sorts keys at init
  @assert issorted(keys(nt1))
  @assert issorted(keys(nt2))
  return intersect_keys(nt1, nt2), intersect_keys(nt2, nt1)
end

categories_diff(nt1::NamedTuple, nt2::NamedTuple) = symdiff_keys(nt1, nt2)

function shared_categories_fusion_rule(shared1::T, shared2::T) where {T<:NamedTuple}
  fused = map(fusion_rule, values(shared1), values(shared2))
  return recover_style(T, fused)
end
