using HalfIntegers: Half, twice

#
# Conventional SU2 group
# using "J" labels
#

struct SU2 <: AbstractCategory
  j::Half{Int}
end

dual(s::SU2) = s

label(s::SU2) = s.j

trivial(::Type{SU2}) = SU2(0)

dimension(s::SU2) = twice(label(s)) + 1

label_fusion_rule(::Type{SU2}, j1, j2) = abs(j1 - j2):(j1 + j2)
