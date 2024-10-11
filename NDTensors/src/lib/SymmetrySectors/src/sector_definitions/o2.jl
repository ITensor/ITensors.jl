#
# Orthogonal group O(2)
# isomorphic to Z_2 ⋉ U(1)
# isomorphic to SU(2) subgroup with Sz conservation + Sz-reversal
#
# O(2) has 3 kinds of irreps:
# - trivial irrep, or "0e", corresponds to Sz=0 and even under Sz-reversal
# - "zero odd", or "0o" irrep, corresponds to Sz=0 and odd under Sz-reversal
# - 2-dimensional Sz=±|m| irrep, with m a half integer
#

using HalfIntegers: Half, HalfInteger
using ..GradedAxes: GradedAxes

# here we use only one half-integer as label:
# - l=0 for trivial
# - l=-1 for zero odd
# - l=+|m| for Sz=±|m|
struct O2 <: AbstractSector
  l::Half{Int}
end

SymmetryStyle(::Type{O2}) = NotAbelianStyle()

sector_label(s::O2) = s.l

trivial(::Type{O2}) = O2(0)
zero_odd(::Type{O2}) = O2(-1)

is_zero_even_or_odd(s::O2) = is_zero_even_or_odd(sector_label(s))
iszero_odd(s::O2) = iszero_odd(sector_label(s))

is_zero_even_or_odd(l::HalfInteger) = iszero_even(l) || iszero_odd(l)
iszero_even(l::HalfInteger) = l == sector_label(trivial(O2))
iszero_odd(l::HalfInteger) = l == sector_label(zero_odd(O2))

quantum_dimension(::NotAbelianStyle, s::O2) = 2 - is_zero_even_or_odd(s)

GradedAxes.dual(s::O2) = s

function Base.show(io::IO, s::O2)
  if iszero_odd(s)
    disp = "0o"
  elseif istrivial(s)
    disp = "0e"
  else
    disp = "±" * string(sector_label(s))
  end
  return print(io, "O(2)[", disp, "]")
end

function label_fusion_rule(::Type{O2}, l1, l2)
  if is_zero_even_or_odd(l1)
    degens = [1]
    if is_zero_even_or_odd(l2)
      labels = l1 == l2 ? [sector_label(trivial(O2))] : [sector_label(zero_odd(O2))]
    else
      labels = [l2]
    end
  else
    if is_zero_even_or_odd(l2)
      degens = [1]
      labels = [l1]
    else
      if l1 == l2
        degens = [1, 1, 1]
        labels = [sector_label(zero_odd(O2)), sector_label(trivial(O2)), 2 * l1]
      else
        degens = [1, 1]
        labels = [abs(l1 - l2), l1 + l2]
      end
    end
  end
  return O2.(labels) .=> degens
end
