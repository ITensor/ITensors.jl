#
# Cyclic group Zₙ
#

using ...GradedAxes: GradedAxes

struct Z{N} <: AbstractCategory
  m::Int
  Z{N}(m) where {N} = new{N}(m % N)
end

SymmetryStyle(::Z) = AbelianStyle()

category_label(c::Z) = c.m
modulus(::Type{Z{N}}) where {N} = N

modulus(c::Z) = modulus(typeof(c))

trivial(category_type::Type{<:Z}) = category_type(0)

function label_fusion_rule(category_type::Type{<:Z}, n1, n2)
  return (n1 + n2) % modulus(category_type)
end

GradedAxes.dual(c::Z) = typeof(c)(mod(-category_label(c), modulus(c)))
