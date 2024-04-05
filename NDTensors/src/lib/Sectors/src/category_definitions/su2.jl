using HalfIntegers: Half, half, twice

#
# Conventional SU2 group
# using "J" labels
#

struct SU2 <: AbstractCategory
  j::Half{Int}
end

SymmetryStyle(::SU2) = NonAbelianGroup()

GradedAxes.dual(s::SU2) = s

category_label(s::SU2) = s.j

trivial(::Type{SU2}) = SU2(0)
fundamental(::Type{SU2}) = SU2(half(1))
adjoint(::Type{SU2}) = SU2(1)

quantum_dimension(::NonAbelianGroup, s::SU2) = twice(category_label(s)) + 1

function label_fusion_rule(::Type{SU2}, j1, j2)
  labels = collect(abs(j1 - j2):(j1 + j2))
  degen = ones(Int, length(labels))
  return degen, labels
end
