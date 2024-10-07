# This files defines a structure for Cartesian product of 2 or more fusion sectors
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

using BlockArrays: blocklengths
using ..LabelledNumbers: LabelledInteger, label, labelled, unlabel
using ..GradedAxes: AbstractGradedUnitRange, GradedAxes, dual

# =====================================  Definition  =======================================
struct SectorProduct{Sectors} <: AbstractSector
  sectors::Sectors
  global _SectorProduct(l) = new{typeof(l)}(l)
end

SectorProduct(c::SectorProduct) = _SectorProduct(sectors(c))

sectors(s::SectorProduct) = s.sectors

# =================================  Sectors interface  ====================================
SymmetryStyle(T::Type{<:SectorProduct}) = sectors_symmetrystyle(sectors_type(T))

function quantum_dimension(::NotAbelianStyle, s::SectorProduct)
  return mapreduce(quantum_dimension, *, sectors(s))
end

# use map instead of broadcast to support both Tuple and NamedTuple
GradedAxes.dual(s::SectorProduct) = SectorProduct(map(dual, sectors(s)))

function trivial(type::Type{<:SectorProduct})
  return SectorProduct(sectors_trivial(sectors_type(type)))
end

# ===================================  Base interface  =====================================
function Base.:(==)(A::SectorProduct, B::SectorProduct)
  return sectors_isequal(sectors(A), sectors(B))
end

function Base.show(io::IO, s::SectorProduct)
  (length(sectors(s)) < 2) && print(io, "sector")
  print(io, "(")
  symbol = ""
  for p in pairs(sectors(s))
    print(io, symbol)
    sector_show(io, p[1], p[2])
    symbol = " × "
  end
  return print(io, ")")
end

sector_show(io::IO, ::Int, v) = print(io, v)
sector_show(io::IO, k::Symbol, v) = print(io, "($k=$v,)")

function Base.isless(s1::SectorProduct, s2::SectorProduct)
  return sectors_isless(sectors(s1), sectors(s2))
end

# =======================================  shared  =========================================
# there are 2 implementations for SectorProduct
# - ordered-like with a Tuple
# - dictionary-like with a NamedTuple

function sectors_isequal(s1, s2)
  return ==(sym_sectors_insert_unspecified(s1, s2)...)
end

# get clean results when mixing implementations
function sectors_isequal(nt::NamedTuple, ::Tuple{})
  return sectors_isequal(nt, (;))
end
function sectors_isequal(::Tuple{}, nt::NamedTuple)
  return sectors_isequal((;), nt)
end
function sectors_isequal(::NamedTuple{()}, t::Tuple)
  return sectors_isequal((), t)
end
function sectors_isequal(t::Tuple, ::NamedTuple{()})
  return sectors_isequal(t, ())
end
sectors_isequal(::Tuple{}, ::NamedTuple{()}) = true
sectors_isequal(::NamedTuple{()}, ::Tuple{}) = true
sectors_isequal(::Tuple, ::NamedTuple) = false
sectors_isequal(::NamedTuple, ::Tuple) = false

function sectors_isless(nt::NamedTuple, ::Tuple{})
  return sectors_isless(nt, (;))
end
function sectors_isless(::Tuple{}, nt::NamedTuple)
  return sectors_isless((;), nt)
end
function sectors_isless(::NamedTuple{()}, t::Tuple)
  return sectors_isless((), t)
end
function sectors_isless(t::Tuple, ::NamedTuple{()})
  return sectors_isless(t, ())
end
function sectors_isless(s1, s2)
  return isless(sym_sectors_insert_unspecified(s1, s2)...)
end

sectors_isless(::NamedTuple, ::Tuple) = throw(ArgumentError("Not implemented"))
sectors_isless(::Tuple, ::NamedTuple) = throw(ArgumentError("Not implemented"))

sectors_type(::Type{<:SectorProduct{T}}) where {T} = T

function sectors_fusion_rule(sects1, sects2)
  shared_sect = shared_sectors_fusion_rule(sectors_common(sects1, sects2)...)
  diff_sect = SectorProduct(sectors_diff(sects1, sects2))
  return shared_sect × diff_sect
end

# edge case with empty sectors
sectors_fusion_rule(sects::Tuple, ::NamedTuple{()}) = SectorProduct(sects)
sectors_fusion_rule(::NamedTuple{()}, sects::Tuple) = SectorProduct(sects)
sectors_fusion_rule(sects::NamedTuple, ::Tuple{}) = SectorProduct(sects)
sectors_fusion_rule(::Tuple{}, sects::NamedTuple) = SectorProduct(sects)

function recover_style(T::Type, fused)
  style = sectors_symmetrystyle(T)
  return recover_sector_product_type(style, T, fused)
end

function recover_sector_product_type(::AbelianStyle, T::Type, fused)
  return recover_sector_product_type(T, fused)
end

function recover_sector_product_type(::NotAbelianStyle, T::Type, fused)
  # here fused contains at least one graded unit range.
  # convert eg. Tuple{GradedUnitRange{SU2}, GradedUnitRange{SU2}} into GradedUnitRange{SU2×SU2}
  g = reduce(×, fused)
  # convention: keep unsorted blocklabels as produced by F order loops in ×
  type_fixed = recover_sector_product_type(T, g)
  return type_fixed
end

function recover_sector_product_type(T::Type, g0::AbstractGradedUnitRange)
  new_labels = recover_sector_product_type.(T, blocklabels(g0))
  new_blocklengths = labelled.(unlabel.(blocklengths(g0)), new_labels)
  return gradedrange(new_blocklengths)
end

function recover_sector_product_type(T::Type, c::AbstractSector)
  return recover_sector_product_type(T, SectorProduct(c))
end

function recover_sector_product_type(T::Type, c::SectorProduct)
  return recover_sector_product_type(T, sectors(c))
end

function recover_sector_product_type(T::Type{<:SectorProduct}, sects)
  return recover_sector_product_type(sectors_type(T), sects)
end

function recover_sector_product_type(T::Type, sects)
  return SectorProduct(T(sects))
end

# =================================  Cartesian Product  ====================================
×(c1::AbstractSector, c2::AbstractSector) = ×(SectorProduct(c1), SectorProduct(c2))
function ×(p1::SectorProduct, p2::SectorProduct)
  return SectorProduct(sectors_product(sectors(p1), sectors(p2)))
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
  return to_gradedrange(sectors_fusion_rule(sectors(s1), sectors(s2)))
end

# Abelian case: fusion returns SectorProduct
function fusion_rule(::AbelianStyle, s1::SectorProduct, s2::SectorProduct)
  return sectors_fusion_rule(sectors(s1), sectors(s2))
end

# lift ambiguities for TrivialSector
fusion_rule(::AbelianStyle, c::SectorProduct, ::TrivialSector) = c
fusion_rule(::AbelianStyle, ::TrivialSector, c::SectorProduct) = c
fusion_rule(::NotAbelianStyle, c::SectorProduct, ::TrivialSector) = to_gradedrange(c)
fusion_rule(::NotAbelianStyle, ::TrivialSector, c::SectorProduct) = to_gradedrange(c)

# ===============================  Ordered implementation  =================================
SectorProduct(t::Tuple) = _SectorProduct(t)
SectorProduct(sects::AbstractSector...) = SectorProduct(sects)

function sectors_symmetrystyle(T::Type{<:Tuple})
  return mapreduce(SymmetryStyle, combine_styles, fieldtypes(T); init=AbelianStyle())
end

sectors_product(::NamedTuple{()}, l1::Tuple) = l1
sectors_product(l2::Tuple, ::NamedTuple{()}) = l2
sectors_product(l1::Tuple, l2::Tuple) = (l1..., l2...)

sectors_trivial(type::Type{<:Tuple}) = trivial.(fieldtypes(type))

function sectors_common(t1::Tuple, t2::Tuple)
  n = min(length(t1), length(t2))
  return t1[begin:n], t2[begin:n]
end

function sectors_diff(t1::Tuple, t2::Tuple)
  n1 = length(t1)
  n2 = length(t2)
  return n1 < n2 ? t2[(n1 + 1):end] : t1[(n2 + 1):end]
end

function shared_sectors_fusion_rule(shared1::T, shared2::T) where {T<:Tuple}
  fused = map(fusion_rule, shared1, shared2)
  return recover_style(T, fused)
end

function sym_sectors_insert_unspecified(t1::Tuple, t2::Tuple)
  n1 = length(t1)
  n2 = length(t2)
  return (t1..., trivial.(t2[(n1 + 1):end])...), (t2..., trivial.(t1[(n2 + 1):end])...)
end

# ===========================  Dictionary-like implementation  =============================
function SectorProduct(nt::NamedTuple)
  sectors = sort_keys(nt)
  return _SectorProduct(sectors)
end

SectorProduct(; kws...) = SectorProduct((; kws...))

function SectorProduct(pairs::Pair...)
  keys = ntuple(n -> Symbol(pairs[n][1]), length(pairs))
  vals = ntuple(n -> pairs[n][2], length(pairs))
  return SectorProduct(NamedTuple{keys}(vals))
end

function sectors_symmetrystyle(NT::Type{<:NamedTuple})
  return mapreduce(SymmetryStyle, combine_styles, fieldtypes(NT); init=AbelianStyle())
end

function sym_sectors_insert_unspecified(nt1::NamedTuple, nt2::NamedTuple)
  return sectors_insert_unspecified(nt1, nt2), sectors_insert_unspecified(nt2, nt1)
end

function sectors_insert_unspecified(nt1::NamedTuple, nt2::NamedTuple)
  diff1 = sectors_trivial(typeof(setdiff_keys(nt2, nt1)))
  return sort_keys(union_keys(nt1, diff1))
end

sectors_product(l1::NamedTuple, ::Tuple{}) = l1
sectors_product(::Tuple{}, l2::NamedTuple) = l2
function sectors_product(l1::NamedTuple, l2::NamedTuple)
  if length(intersect_keys(l1, l2)) > 0
    throw(ArgumentError("Cannot define product of shared keys"))
  end
  return union_keys(l1, l2)
end

function sectors_trivial(type::Type{<:NamedTuple{Keys}}) where {Keys}
  return NamedTuple{Keys}(trivial.(fieldtypes(type)))
end

function sectors_common(nt1::NamedTuple, nt2::NamedTuple)
  # SectorProduct(nt::NamedTuple) sorts keys at init
  @assert issorted(keys(nt1))
  @assert issorted(keys(nt2))
  return intersect_keys(nt1, nt2), intersect_keys(nt2, nt1)
end

sectors_diff(nt1::NamedTuple, nt2::NamedTuple) = symdiff_keys(nt1, nt2)

function shared_sectors_fusion_rule(shared1::T, shared2::T) where {T<:NamedTuple}
  fused = map(fusion_rule, values(shared1), values(shared2))
  return recover_style(T, fused)
end
