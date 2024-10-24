# This files defines a structure for Cartesian product of 2 or more fusion sectors
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

using BlockArrays: blocklengths
using ..LabelledNumbers: LabelledInteger, label, labelled, unlabel
using ..GradedAxes: AbstractGradedUnitRange, GradedAxes, dual

# =====================================  Definition  =======================================
struct SectorProduct{Sectors} <: AbstractSector
  product_sectors::Sectors
  global _SectorProduct(l) = new{typeof(l)}(l)
end

SectorProduct(c::SectorProduct) = _SectorProduct(product_sectors(c))

product_sectors(s::SectorProduct) = s.product_sectors

# =================================  Sectors interface  ====================================
function SymmetryStyle(T::Type{<:SectorProduct})
  return product_sectors_symmetrystyle(product_sectors_type(T))
end

function quantum_dimension(::NotAbelianStyle, s::SectorProduct)
  return mapreduce(quantum_dimension, *, product_sectors(s))
end

# use map instead of broadcast to support both Tuple and NamedTuple
GradedAxes.dual(s::SectorProduct) = SectorProduct(map(dual, product_sectors(s)))

function trivial(type::Type{<:SectorProduct})
  return SectorProduct(product_sectors_trivial(product_sectors_type(type)))
end

# ===================================  Base interface  =====================================
function Base.:(==)(A::SectorProduct, B::SectorProduct)
  return product_sectors_isequal(product_sectors(A), product_sectors(B))
end

function Base.show(io::IO, s::SectorProduct)
  (length(product_sectors(s)) < 2) && print(io, "sector")
  print(io, "(")
  symbol = ""
  for p in pairs(product_sectors(s))
    print(io, symbol)
    sector_show(io, p[1], p[2])
    symbol = " × "
  end
  return print(io, ")")
end

sector_show(io::IO, ::Int, v) = print(io, v)
sector_show(io::IO, k::Symbol, v) = print(io, "($k=$v,)")

function Base.isless(s1::SectorProduct, s2::SectorProduct)
  return product_sectors_isless(product_sectors(s1), product_sectors(s2))
end

# =======================================  shared  =========================================
# there are 2 implementations for SectorProduct
# - ordered-like with a Tuple
# - dictionary-like with a NamedTuple

function sym_product_sectors_insert_unspecified(s1, s2)
  return product_sectors_insert_unspecified(s1, s2),
  product_sectors_insert_unspecified(s2, s1)
end

function product_sectors_isequal(s1, s2)
  return ==(sym_product_sectors_insert_unspecified(s1, s2)...)
end

# get clean results when mixing implementations
product_sectors_isequal(nt::NamedTuple, t::Tuple) = product_sectors_isequal(t, nt)
function product_sectors_isequal(t::Tuple, nt::NamedTuple)
  if isempty(t)
    return product_sectors_isequal((;), nt)
  elseif isempty(nt)
    return product_sectors_isequal(t, ())
  end
  return false
end

function product_sectors_isless(nt::NamedTuple, ::Tuple{})
  return product_sectors_isless(nt, (;))
end
function product_sectors_isless(::Tuple{}, nt::NamedTuple)
  return product_sectors_isless((;), nt)
end
function product_sectors_isless(::NamedTuple{()}, t::Tuple)
  return product_sectors_isless((), t)
end
function product_sectors_isless(t::Tuple, ::NamedTuple{()})
  return product_sectors_isless(t, ())
end
function product_sectors_isless(s1, s2)
  return isless(sym_product_sectors_insert_unspecified(s1, s2)...)
end

product_sectors_isless(::NamedTuple, ::Tuple) = throw(ArgumentError("Not implemented"))
product_sectors_isless(::Tuple, ::NamedTuple) = throw(ArgumentError("Not implemented"))

product_sectors_type(::Type{<:SectorProduct{T}}) where {T} = T

function product_sectors_fusion_rule(sects1, sects2)
  isempty(sects1) && return SectorProduct(sects2)
  isempty(sects2) && return SectorProduct(sects1)
  shared_sect = shared_product_sectors_fusion_rule(
    product_sectors_common(sects1, sects2)...
  )
  diff_sect = SectorProduct(product_sectors_diff(sects1, sects2))
  return shared_sect × diff_sect
end

# =================================  Cartesian Product  ====================================
×(c1::AbstractSector, c2::AbstractSector) = ×(SectorProduct(c1), SectorProduct(c2))
function ×(p1::SectorProduct, p2::SectorProduct)
  return SectorProduct(product_sectors_product(product_sectors(p1), product_sectors(p2)))
end

×(a, g::AbstractUnitRange) = ×(to_gradedrange(a), g)
×(g::AbstractUnitRange, b) = ×(g, to_gradedrange(b))
×(nt1::NamedTuple, nt2::NamedTuple) = ×(SectorProduct(nt1), SectorProduct(nt2))
×(c1::NamedTuple, c2::AbstractSector) = ×(SectorProduct(c1), SectorProduct(c2))
×(c1::AbstractSector, c2::NamedTuple) = ×(SectorProduct(c1), SectorProduct(c2))

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
# cast AbstractSector to SectorProduct
function fusion_rule(style::SymmetryStyle, c1::SectorProduct, c2::AbstractSector)
  return fusion_rule(style, c1, SectorProduct(c2))
end
function fusion_rule(style::SymmetryStyle, c1::AbstractSector, c2::SectorProduct)
  return fusion_rule(style, SectorProduct(c1), c2)
end

# generic case: fusion returns a GradedAxes, even for fusion with Empty
function fusion_rule(::NotAbelianStyle, s1::SectorProduct, s2::SectorProduct)
  return to_gradedrange(
    product_sectors_fusion_rule(product_sectors(s1), product_sectors(s2))
  )
end

# Abelian case: fusion returns SectorProduct
function fusion_rule(::AbelianStyle, s1::SectorProduct, s2::SectorProduct)
  return label(only(fusion_rule(NotAbelianStyle(), s1, s2)))
end

# lift ambiguities for TrivialSector
fusion_rule(::AbelianStyle, c::SectorProduct, ::TrivialSector) = c
fusion_rule(::AbelianStyle, ::TrivialSector, c::SectorProduct) = c
fusion_rule(::NotAbelianStyle, c::SectorProduct, ::TrivialSector) = to_gradedrange(c)
fusion_rule(::NotAbelianStyle, ::TrivialSector, c::SectorProduct) = to_gradedrange(c)

# ===============================  Ordered implementation  =================================
SectorProduct(t::Tuple) = _SectorProduct(t)
SectorProduct(sects::AbstractSector...) = SectorProduct(sects)

function product_sectors_symmetrystyle(T::Type{<:Tuple})
  return mapreduce(SymmetryStyle, combine_styles, fieldtypes(T); init=AbelianStyle())
end

product_sectors_product(::NamedTuple{()}, l1::Tuple) = l1
product_sectors_product(l2::Tuple, ::NamedTuple{()}) = l2
product_sectors_product(l1::Tuple, l2::Tuple) = (l1..., l2...)

product_sectors_trivial(T::Type{<:Tuple}) = trivial.(fieldtypes(T))

function product_sectors_common(t1::Tuple, t2::Tuple)
  n = min(length(t1), length(t2))
  return t1[begin:n], t2[begin:n]
end

function product_sectors_diff(t1::Tuple, t2::Tuple)
  n1 = length(t1)
  n2 = length(t2)
  return n1 < n2 ? t2[(n1 + 1):end] : t1[(n2 + 1):end]
end

function shared_product_sectors_fusion_rule(shared1::T, shared2::T) where {T<:Tuple}
  return mapreduce(
    to_gradedrange ∘ fusion_rule,
    ×,
    shared1,
    shared2;
    init=to_gradedrange(SectorProduct(())),
  )
end

function product_sectors_insert_unspecified(t1::Tuple, t2::Tuple)
  n1 = length(t1)
  return (t1..., trivial.(t2[(n1 + 1):end])...)
end

# ===========================  Dictionary-like implementation  =============================
function SectorProduct(nt::NamedTuple)
  product_sectors = sort_keys(nt)
  return _SectorProduct(product_sectors)
end

SectorProduct(; kws...) = SectorProduct((; kws...))

function SectorProduct(pairs::Pair...)
  keys = Symbol.(first.(pairs))
  vals = last.(pairs)
  return SectorProduct(NamedTuple{keys}(vals))
end

function product_sectors_symmetrystyle(NT::Type{<:NamedTuple})
  return mapreduce(SymmetryStyle, combine_styles, fieldtypes(NT); init=AbelianStyle())
end

function product_sectors_insert_unspecified(nt1::NamedTuple, nt2::NamedTuple)
  diff1 = product_sectors_trivial(typeof(setdiff_keys(nt2, nt1)))
  return sort_keys(union_keys(nt1, diff1))
end

product_sectors_product(l1::NamedTuple, ::Tuple{}) = l1
product_sectors_product(::Tuple{}, l2::NamedTuple) = l2
function product_sectors_product(l1::NamedTuple, l2::NamedTuple)
  if length(intersect_keys(l1, l2)) > 0
    throw(ArgumentError("Cannot define product of shared keys"))
  end
  return union_keys(l1, l2)
end

function product_sectors_trivial(NT::Type{<:NamedTuple{Keys}}) where {Keys}
  return NamedTuple{Keys}(trivial.(fieldtypes(NT)))
end

function product_sectors_common(nt1::NamedTuple, nt2::NamedTuple)
  # SectorProduct(nt::NamedTuple) sorts keys at init
  @assert issorted(keys(nt1))
  @assert issorted(keys(nt2))
  return intersect_keys(nt1, nt2), intersect_keys(nt2, nt1)
end

product_sectors_diff(nt1::NamedTuple, nt2::NamedTuple) = symdiff_keys(nt1, nt2)

function map_blocklabels(f, r::AbstractUnitRange)
  return gradedrange(labelled.(unlabel.(blocklengths(r)), f.(blocklabels(r))))
end

function shared_product_sectors_fusion_rule(shared1::NT, shared2::NT) where {NT<:NamedTuple}
  tuple_fused = shared_product_sectors_fusion_rule(values(shared1), values(shared2))
  return map_blocklabels(SectorProduct ∘ NT ∘ product_sectors ∘ SectorProduct, tuple_fused)
end
