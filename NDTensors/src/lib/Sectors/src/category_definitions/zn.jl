using HalfIntegers: Half

#
# Cyclic group Zₙ
#

struct Z{N} <: AbstractCategory
  m::Half{Int}
  Z{N}(m) where {N} = new{N}(m % N)
end

label(c::Z) = c.m
modulus(::Type{Z{N}}) where {N} = N

modulus(c::Z) = modulus(typeof(c))

dimension(::Z) = 1

trivial(category_type::Type{<:Z}) = category_type(0)

label_fusion_rule(category_type::Type{<:Z}, n1, n2) = ((n1 + n2) % modulus(category_type),)

dual(c::Z) = typeof(c)(mod(-label(c), modulus(c)))
