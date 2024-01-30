using HalfIntegers: Half

#
# U₁ group (circle group, or particle number, total Sz etc.)
#

struct U1 <: AbstractCategory
  n::Half{Int}
end

dual(u::U1) = U1(-u.n)

label(u::U1) = u.n

dimension(::U1) = 1

trivial(::Type{U1}) = U1(0)

label_fusion_rule(::Type{U1}, n1, n2) = (n1 + n2,)
