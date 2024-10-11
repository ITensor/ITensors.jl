#
# Quantum 'group' su2ₖ
#

using HalfIntegers: Half
using ...GradedAxes: GradedAxes

struct su2{k} <: AbstractSector
  j::Half{Int}
end

SymmetryStyle(::Type{<:su2}) = NotAbelianStyle()

GradedAxes.dual(s::su2) = s

sector_label(s::su2) = s.j

level(::su2{k}) where {k} = k

trivial(::Type{su2{k}}) where {k} = su2{k}(0)

function label_fusion_rule(::Type{su2{k}}, j1, j2) where {k}
  labels = collect(abs(j1 - j2):min(k - j1 - j2, j1 + j2))
  degen = ones(Int, length(labels))
  sectors = su2{k}.(labels)
  return sectors .=> degen
end
