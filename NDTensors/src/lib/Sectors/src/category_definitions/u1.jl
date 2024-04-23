#
# U₁ group (circle group, or particle number, total Sz etc.)
#

# Parametric type to allow both integer label as well as
# HalfInteger for easy conversion to/from SU(2)
struct U1{T} <: AbstractCategory
  n::T
end

SymmetryStyle(::U1) = AbelianGroup()

GradedAxes.dual(u::U1) = U1(-u.n)

category_label(u::U1) = u.n

trivial(::Type{U1}) = trivial(U1{Int})
trivial(::Type{U1{T}}) where {T} = U1(T(0))

label_fusion_rule(::Type{<:U1}, n1, n2) = n1 + n2

# hide label type in printing
function Base.show(io::IO, u::U1)
  return print(io, "U(1)[", category_label(u), "]")
end
