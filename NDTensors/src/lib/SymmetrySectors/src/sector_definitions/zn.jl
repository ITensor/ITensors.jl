#
# Cyclic group Zₙ
#

using ...GradedAxes: GradedAxes

struct Z{N} <: AbstractSector
  m::Int
  Z{N}(m) where {N} = new{N}(m % N)
end

SymmetryStyle(::Type{<:Z}) = AbelianStyle()

sector_label(c::Z) = c.m
modulus(::Type{Z{N}}) where {N} = N

modulus(c::Z) = modulus(typeof(c))

trivial(sector_type::Type{<:Z}) = sector_type(0)

function abelian_label_fusion_rule(sector_type::Type{<:Z}, n1, n2)
  return sector_type((n1 + n2) % modulus(sector_type))
end

GradedAxes.dual(c::Z) = typeof(c)(mod(-sector_label(c), modulus(c)))
