using HalfIntegers: Half

#
# Quantum 'group' su2ₖ
#

struct su2{k} <: AbstractCategory
  j::Half{Int}
end

dual(s::su2) = s

label(s::su2) = s.j

level(s::su2{k}) where {k} = k

trivial(::Type{su2{k}}) where {k} = su2{k}(0)

function label_fusion_rule(::Type{su2{k}}, j1, j2) where {k}
  return abs(j1-j2):min(k-j1-j2, j1+j2)
end
