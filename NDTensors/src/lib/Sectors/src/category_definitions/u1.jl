using HalfIntegers: Half

#
# U₁ group (circle group, or particle number, total Sz etc.)
#

# use HalfInteger as internal label to allow easy conversion to/from SU(2)
# still allows to call U1(::Int) and to use it as an Int
struct U1 <: AbstractCategory
  n::Half{Int}
end

SymmetryStyle(::U1) = AbelianGroup()

GradedAxes.dual(u::U1) = U1(-u.n)

category_label(u::U1) = u.n

trivial(::Type{U1}) = U1(0)

label_fusion_rule(::Type{U1}, n1, n2) = n1 + n2
