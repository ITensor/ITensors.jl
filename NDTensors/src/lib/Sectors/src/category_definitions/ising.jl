using HalfIntegers: HalfIntegers

#
# Ising category
#
# (same fusion rules as su2{2})
#

struct Ising <: AbstractCategory
  l::HalfIntegers.Half{Int}
end

# TODO: Use `Val` dispatch here?
function Ising(s::AbstractString)
  for (a, v) in enumerate(("1", "σ", "ψ"))
    (v == s) && return Ising((a - 1)//2)
  end
  return error("Unrecognized input \"$s\" to Ising constructor")
end

SymmetryStyle(::Ising) = NonGroupCategory()

GradedAxes.dual(i::Ising) = i

category_label(i::Ising) = i.l

trivial(::Type{Ising}) = Ising(0)

quantum_dimension(::NonGroupCategory, i::Ising) = (category_label(i) == 1//2) ? √2 : 1.0

# Fusion rules identical to su2₂
label_fusion_rule(::Type{Ising}, l1, l2) = label_fusion_rule(su2{2}, l1, l2)

# TODO: Use `Val` dispatch here?
label_to_str(i::Ising) = ("1", "σ", "ψ")[HalfIntegers.twice(category_label(i)) + 1]

function Base.show(io::IO, f::Ising)
  return print(io, "Ising(", label_to_str(f), ")")
end
