# This files defines a structure for Cartesian product of 2 or more fusion sectors
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

using BlockArrays: blocklengths
using ..LabelledNumbers: LabelledInteger, label, labelled, unlabel
using ..GradedAxes: AbstractGradedUnitRange, GradedAxes, dual

# =====================================  Definition  =======================================
struct SectorProduct{Sectors} <: AbstractSector
  arguments::Sectors
  global _SectorProduct(l) = new{typeof(l)}(l)
end

SectorProduct(c::SectorProduct) = _SectorProduct(arguments(c))

arguments(s::SectorProduct) = s.arguments

# =================================  Sectors interface  ====================================
function SymmetryStyle(T::Type{<:SectorProduct})
  return arguments_symmetrystyle(arguments_type(T))
end

function quantum_dimension(::NotAbelianStyle, s::SectorProduct)
  return mapreduce(quantum_dimension, *, arguments(s))
end

# use map instead of broadcast to support both Tuple and NamedTuple
GradedAxes.dual(s::SectorProduct) = SectorProduct(map(dual, arguments(s)))

function trivial(type::Type{<:SectorProduct})
  return SectorProduct(arguments_trivial(arguments_type(type)))
end

# ===================================  Base interface  =====================================
function Base.:(==)(A::SectorProduct, B::SectorProduct)
  return arguments_isequal(arguments(A), arguments(B))
end

function Base.show(io::IO, s::SectorProduct)
  (length(arguments(s)) < 2) && print(io, "sector")
  print(io, "(")
  symbol = ""
  for p in pairs(arguments(s))
    print(io, symbol)
    sector_show(io, p[1], p[2])
    symbol = " × "
  end
  return print(io, ")")
end

sector_show(io::IO, ::Int, v) = print(io, v)
sector_show(io::IO, k::Symbol, v) = print(io, "($k=$v,)")

function Base.isless(s1::SectorProduct, s2::SectorProduct)
  return arguments_isless(arguments(s1), arguments(s2))
end

# =======================================  shared  =========================================
# there are 2 implementations for SectorProduct
# - ordered-like with a Tuple
# - dictionary-like with a NamedTuple

arguments_type(::Type{<:SectorProduct{T}}) where {T} = T

arguments_maybe_insert_unspecified(s1, ::Any) = s1
function sym_arguments_maybe_insert_unspecified(s1, s2)
  return arguments_maybe_insert_unspecified(s1, s2),
  arguments_maybe_insert_unspecified(s2, s1)
end

function make_empty_match(a1, b1)
  a2 = isempty(a1) ? empty(b1) : a1
  b2 = isempty(b1) ? empty(a2) : b1
  return a2, b2
end

function arguments_isequal(a1, b1)
  return ==(sym_arguments_maybe_insert_unspecified(make_empty_match(a1, b1)...)...)
end

function arguments_product(s1, s2)
  isempty(s1) && return s2
  isempty(s2) && return s1
  return throw(ArgumentError("Mixing non-empty storage types is illegal"))
end

function arguments_isless(a1, b1)
  return isless(sym_arguments_maybe_insert_unspecified(make_empty_match(a1, b1)...)...)
end

# =================================  Cartesian Product  ====================================
×(c1::AbstractSector, c2::AbstractSector) = ×(SectorProduct(c1), SectorProduct(c2))
function ×(p1::SectorProduct, p2::SectorProduct)
  return SectorProduct(arguments_product(arguments(p1), arguments(p2)))
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
  return to_gradedrange(arguments_fusion_rule(arguments(s1), arguments(s2)))
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

function arguments_fusion_rule(sects1, sects2)
  isempty(sects1) && return SectorProduct(sects2)
  isempty(sects2) && return SectorProduct(sects1)
  shared_sect = shared_arguments_fusion_rule(arguments_common(sects1, sects2)...)
  diff_sect = SectorProduct(arguments_diff(sects1, sects2))
  return shared_sect × diff_sect
end

# ===============================  Ordered implementation  =================================
SectorProduct(t::Tuple) = _SectorProduct(t)
SectorProduct(sects::AbstractSector...) = SectorProduct(sects)

function arguments_symmetrystyle(T::Type{<:Tuple})
  return mapreduce(SymmetryStyle, combine_styles, fieldtypes(T); init=AbelianStyle())
end

arguments_product(l1::Tuple, l2::Tuple) = (l1..., l2...)

arguments_trivial(T::Type{<:Tuple}) = trivial.(fieldtypes(T))

function arguments_common(t1::Tuple, t2::Tuple)
  n = min(length(t1), length(t2))
  return t1[begin:n], t2[begin:n]
end

function arguments_diff(t1::Tuple, t2::Tuple)
  n1 = length(t1)
  n2 = length(t2)
  return n1 < n2 ? t2[(n1 + 1):end] : t1[(n2 + 1):end]
end

function shared_arguments_fusion_rule(shared1::T, shared2::T) where {T<:Tuple}
  return mapreduce(
    to_gradedrange ∘ fusion_rule,
    ×,
    shared1,
    shared2;
    init=to_gradedrange(SectorProduct(())),
  )
end

function arguments_maybe_insert_unspecified(t1::Tuple, t2::Tuple)
  n1 = length(t1)
  return (t1..., trivial.(t2[(n1 + 1):end])...)
end

# ===========================  Dictionary-like implementation  =============================
function SectorProduct(nt::NamedTuple)
  arguments = sort_keys(nt)
  return _SectorProduct(arguments)
end

SectorProduct(; kws...) = SectorProduct((; kws...))

function SectorProduct(pairs::Pair...)
  keys = Symbol.(first.(pairs))
  vals = last.(pairs)
  return SectorProduct(NamedTuple{keys}(vals))
end

function arguments_symmetrystyle(NT::Type{<:NamedTuple})
  return mapreduce(SymmetryStyle, combine_styles, fieldtypes(NT); init=AbelianStyle())
end

function arguments_maybe_insert_unspecified(nt1::NamedTuple, nt2::NamedTuple)
  diff1 = arguments_trivial(typeof(setdiff_keys(nt2, nt1)))
  return sort_keys(union_keys(nt1, diff1))
end

function arguments_product(l1::NamedTuple, l2::NamedTuple)
  if length(intersect_keys(l1, l2)) > 0
    throw(ArgumentError("Cannot define product of shared keys"))
  end
  return union_keys(l1, l2)
end

function arguments_trivial(NT::Type{<:NamedTuple{Keys}}) where {Keys}
  return NamedTuple{Keys}(trivial.(fieldtypes(NT)))
end

function arguments_common(nt1::NamedTuple, nt2::NamedTuple)
  # SectorProduct(nt::NamedTuple) sorts keys at init
  @assert issorted(keys(nt1))
  @assert issorted(keys(nt2))
  return intersect_keys(nt1, nt2), intersect_keys(nt2, nt1)
end

arguments_diff(nt1::NamedTuple, nt2::NamedTuple) = symdiff_keys(nt1, nt2)

function map_blocklabels(f, r::AbstractUnitRange)
  return gradedrange(labelled.(unlabel.(blocklengths(r)), f.(blocklabels(r))))
end

function shared_arguments_fusion_rule(shared1::NT, shared2::NT) where {NT<:NamedTuple}
  tuple_fused = shared_arguments_fusion_rule(values(shared1), values(shared2))
  return map_blocklabels(SectorProduct ∘ NT ∘ arguments ∘ SectorProduct, tuple_fused)
end
