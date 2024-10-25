#
# U₁ group (circle group, or particle number, total Sz etc.)
#

using ...GradedAxes: GradedAxes

# Parametric type to allow both integer label as well as
# HalfInteger for easy conversion to/from SU(2)
struct U1{T} <: AbstractSector
  n::T
end

SymmetryStyle(::Type{<:U1}) = AbelianStyle()
sector_label(u::U1) = u.n

set_sector_label(s::U1, sector_label) = typeof(s)(sector_label)
GradedAxes.dual(s::U1) = set_sector_label(s, -sector_label(s))

trivial(::Type{U1}) = trivial(U1{Int})
trivial(::Type{U1{T}}) where {T} = U1(zero(T))

abelian_label_fusion_rule(sector_type::Type{<:U1}, n1, n2) = sector_type(n1 + n2)

# hide label type in printing
function Base.show(io::IO, u::U1)
  return print(io, "U(1)[", sector_label(u), "]")
end

# enforce U1(Int32(1)) == U1(1)
Base.:(==)(s1::U1, s2::U1) = sector_label(s1) == sector_label(s2)
Base.isless(s1::U1, s2::U1) = sector_label(s1) < sector_label(s2)
