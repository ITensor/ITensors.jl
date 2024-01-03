import HalfIntegers: Half

#
# Quantum 'group' su2ₖ
#

struct su2{k} <: AbstractCategory
  j::Half{Int}
end

label(s::su2) = s.j

level(s::su2{k}) where {k} = k

trivial(::Type{su2{k}}) where {k} = su2{k}(0)

function label_fusion_rule(::Type{su2{k}}, j1, j2) where {k}
  return abs(j1 - j2):min(k - j1 - j2, j1 + j2)
end

#
# Fibonacci category
#
# (same fusion rules as subcategory {0,1} of su2{3})
#

struct Fib <: AbstractCategory
  l::Int
end

function Fib(s::AbstractString)
  if s == "1"
    return Fib(0)
  elseif s == "τ"
    return Fib(1)
  end
  return error("Unrecognized input \"$s\" to Fib constructor")
end

label(f::Fib) = f.l

trivial(::Type{Fib}) = Fib(0)

dimension(f::Fib) = istrivial(f) ? 1 : ((1 + √5) / 2)

# Fusion rules identical to su2₃
label_fusion_rule(::Type{Fib}, l1, l2) = label_fusion_rule(su2{3}, l1, l2)

label_to_str(f::Fib) = istrivial(f) ? "1" : "τ"

function Base.show(io::IO, f::Fib)
  return print(io, "Fib(", label_to_str(f), ")")
end

#
# Ising category
#
# (same fusion rules as su2{2})
#

struct Ising <: AbstractCategory
  l::Half{Int}
end

function Ising(s::AbstractString)
  for (a, v) in enumerate(("1", "σ", "ψ"))
    (v == s) && return Ising((a - 1)//2)
  end
  return error("Unrecognized input \"$s\" to Ising constructor")
end

label(i::Ising) = i.l

trivial(::Type{Ising}) = Ising(0)

dimension(i::Ising) = (label(i) == 1//2) ? √2 : 1

# Fusion rules identical to su2₂
label_fusion_rule(::Type{Ising}, l1, l2) = label_fusion_rule(su2{2}, l1, l2)

label_to_str(i::Ising) = ("1", "σ", "ψ")[Int(2 * label(i) + 1)]

function Base.show(io::IO, f::Ising)
  return print(io, "Ising(", label_to_str(f), ")")
end
