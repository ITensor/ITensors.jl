#
# Cyclic group Zₙ
#

using ...GradedAxes: GradedAxes

struct Z{N} <: AbstractSector
  m::Int
  Z{N}(m) where {N} = new{N}(mod(m, N))
end

modulus(::Type{Z{N}}) where {N} = N
modulus(c::Z) = modulus(typeof(c))

SymmetryStyle(::Type{<:Z}) = AbelianStyle()
sector_label(c::Z) = c.m

set_sector_label(s::Z, sector_label) = typeof(s)(sector_label)
GradedAxes.dual(s::Z) = set_sector_label(s, -sector_label(s))

trivial(sector_type::Type{<:Z}) = sector_type(0)

function abelian_label_fusion_rule(sector_type::Type{<:Z}, n1, n2)
  return sector_type(n1 + n2)
end
