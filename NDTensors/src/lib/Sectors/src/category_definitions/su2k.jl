#
# Quantum 'group' su2ₖ
#

using HalfIntegers: Half
using ...GradedAxes: GradedAxes

struct su2{k} <: AbstractCategory
  j::Half{Int}
end

SymmetryStyle(::su2) = NotAbelianStyle()

GradedAxes.dual(s::su2) = s

category_label(s::su2) = s.j

level(::su2{k}) where {k} = k

trivial(::Type{su2{k}}) where {k} = su2{k}(0)

function label_fusion_rule(::Type{su2{k}}, j1, j2) where {k}
  labels = collect(abs(j1 - j2):min(k - j1 - j2, j1 + j2))
  degen = ones(Int, length(labels))
  return degen, labels
end
