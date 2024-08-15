#
# Orthogonal group O(2)
# isomorphic to Z_2 ⋉ U(1)
# corresponds to to SU(2) subgroup with Sz conservation + Sz-reversal
#

struct O2 <: AbstractCategory
  l::HalfIntegers.Half{Int}
end

SymmetryStyle(::O2) = NonAbelianGroup()

category_label(s::O2) = s.l

trivial(::Type{O2}) = O2(0)
zero_odd(::Type{O2}) = O2(-1)

_iszero(s::O2) = _iszero(category_label(s))
_iszero_even(s::O2) = _iszero_even(category_label(s))
_iszero_odd(s::O2) = _iszero_odd(category_label(s))

_iszero(l::HalfIntegers.HalfInteger) = _iszero_even(l) || _iszero_odd(l)
_iszero_even(l::HalfIntegers.HalfInteger) = l == category_label(trivial(O2))
_iszero_odd(l::HalfIntegers.HalfInteger) = l == category_label(zero_odd(O2))

quantum_dimension(::NonAbelianGroup, s::O2) = 2 - _iszero(s)

GradedAxes.dual(s::O2) = s

function Base.show(io::IO, s::O2)
  if _iszero_odd(s)
    disp = "0o"
  elseif _iszero_even(s)
    disp = "0e"
  else
    disp = "±" * string(category_label(s))
  end
  return print(io, "O(2)[", disp, "]")
end

function label_fusion_rule(::Type{O2}, l1, l2)
  if _iszero(l1)
    degens = [1]
    if _iszero(l2)
      labels = l1 == l2 ? [category_label(trivial(O2))] : [category_label(zero_odd(O2))]
    else
      labels = [l2]
    end
  else
    if _iszero(l2)
      degens = [1]
      labels = [l1]
    else
      if l1 == l2
        degens = [1, 1, 1]
        labels = [category_label(zero_odd(O2)), category_label(trivial(O2)), 2 * l1]
      else
        degens = [1, 1]
        labels = [abs(l1 - l2), l1 + l2]
      end
    end
  end
  return degens, labels
end
