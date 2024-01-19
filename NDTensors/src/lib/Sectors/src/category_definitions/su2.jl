using HalfIntegers: Half

#
# Conventional SU2 group
# using "J" labels
#

struct SU2 <: AbstractCategory
  j::Half{Int}
end

label(s::SU2) = s.j

trivial(::Type{SU2}) = SU2(0)

dimension(s::SU2) = 2 * label(s) + 1

label_fusion_rule(::Type{SU2}, j1, j2) = abs(j1 - j2):(j1 + j2)
