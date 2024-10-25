#
# Fibonacci category
#
# (same fusion rules as subcategory {0,1} of su2{3})
#
using ..GradedAxes: GradedAxes

struct Fib <: AbstractSector
  l::Int
end

# TODO: Use `Val` dispatch here?
function Fib(s::AbstractString)
  if s == "1"
    return Fib(0)
  elseif s == "τ"
    return Fib(1)
  end
  return error("Unrecognized input \"$s\" to Fib constructor")
end

SymmetryStyle(::Type{Fib}) = NotAbelianStyle()

GradedAxes.dual(f::Fib) = f

sector_label(f::Fib) = f.l

trivial(::Type{Fib}) = Fib(0)

quantum_dimension(::NotAbelianStyle, f::Fib) = istrivial(f) ? 1.0 : ((1 + √5) / 2)

# Fusion rules identical to su2₃
function label_fusion_rule(::Type{Fib}, l1, l2)
  suk_sectors_degen = label_fusion_rule(su2{3}, l1, l2)
  suk_sectors = first.(suk_sectors_degen)
  degen = last.(suk_sectors_degen)
  sectors = Fib.(sector_label.(suk_sectors))
  return sectors .=> degen
end

label_to_str(f::Fib) = istrivial(f) ? "1" : "τ"

function Base.show(io::IO, f::Fib)
  return print(io, "Fib(", label_to_str(f), ")")
end
