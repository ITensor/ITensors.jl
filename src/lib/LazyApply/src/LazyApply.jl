module LazyApply

import Base:
  ==,
  +,
  -,
  *,
  /,
  ^,
  exp,
  adjoint,
  copy,
  show,
  getindex,
  length,
  isless,
  iterate,
  firstindex,
  lastindex,
  keys,
  reverse,
  size

export Applied, Scaled, Sum, Prod, Exp, coefficient, argument, expand, materialize, terms

struct Applied{F,Args<:Tuple,Kwargs<:NamedTuple}
  f::F
  args::Args
  kwargs::Kwargs
end
Applied(f, args::Tuple) = Applied(f, args, (;))

materialize(x) = x
function materialize(a::Applied)
  return a.f(materialize.(a.args)...; a.kwargs...)
end

function (a1::Applied == a2::Applied)
  return a1.f == a2.f && a1.args == a2.args && a1.kwargs == a2.kwargs
end

#
# Applied algebra
#

# Used for dispatch
const Scaled{C<:Number,A} = Applied{typeof(*),Tuple{C,A},NamedTuple{(),Tuple{}}}
const Sum{A} = Applied{typeof(sum),Tuple{Vector{A}},NamedTuple{(),Tuple{}}}
const Prod{A} = Applied{typeof(prod),Tuple{Vector{A}},NamedTuple{(),Tuple{}}}

# Some convenient empty constructors
Sum{A}() where {A} = Applied(sum, (A[],))
Prod{A}() where {A} = Applied(prod, (A[],))

coefficient(co::Scaled{C}) where {C} = co.args[1]
argument(co::Scaled{C}) where {C} = co.args[2]

#
# Generic algebra
#

# 1.3 * Op("X", 1) + 1.3 * Op("X", 2)
# 1.3 * Op("X", 1) * Op("X", 2) + 1.3 * Op("X", 3) * Op("X", 4)
function (a1::Scaled{C,A} + a2::Scaled{C,A}) where {C,A}
  return Sum{Scaled{C,A}}() + a1 + a2
end

function (a1::Prod{A} + a2::Prod{A}) where {A}
  return Sum{Prod{A}}() + a1 + a2
end

(c::Number * a::Scaled{C}) where {C} = (c * coefficient(a)) * argument(a)
(a::Scaled{C} * c::Number) where {C} = (coefficient(a) * c) * argument(a)

-(a::Scaled{C}) where {C} = (-one(C) * a)
-(a::Sum) = (-1 * a)
-(a::Prod) = (-1 * a)

(os::Sum{A} + o::A) where {A} = Applied(sum, (vcat(os.args[1], [o]),))
(o::A + os::Sum{A}) where {A} = Applied(sum, (vcat([o], os.args[1]),))

(a1::Sum{A} - a2::A) where {A} = a1 + (-a2)
(a1::A - a2::Sum{A}) where {A} = a1 + (-a2)

(a1::Sum{A} - a2::Prod{A}) where {A} = a1 + (-a2)
(a1::Sum{A} - a2::Scaled{C,Prod{A}}) where {C,A} = a1 + (-a2)
(a1::Sum{A} - a2::Sum{Scaled{C,Prod{A}}}) where {C,A} = a1 + (-a2)

(a1::Prod{A} * a2::A) where {A} = Applied(prod, (vcat(only(a1.args), [a2]),))
(a1::A * a2::Prod{A}) where {A} = Applied(prod, (vcat([a1], only(a2.args)),))

# Fixes ambiguity error with:
# *(a1::Applied, a2::Sum)
# *(os::Prod{A}, o::A)
(a1::Prod{Sum{A}} * a2::Sum{A}) where {A} = Applied(prod, (vcat(only(a1.args), [a2]),))

# 1.3 * Op("X", 1) + 1 * Op("X", 2)
# 1.3 * Op("X", 1) * Op("X", 2) + 1 * Op("X", 3)
# 1.3 * Op("X", 1) * Op("X", 2) + 1 * Op("X", 3) * Op("X", 4)
function (co1::Scaled{C1,A} + co2::Scaled{C2,A}) where {C1,C2,A}
  c1, c2 = promote(coefficient(co1), coefficient(co2))
  return c1 * argument(co1) + c2 * argument(co2)
end

# (1.3 * Op("X", 1)) * (1.3 * Op("X", 2))
function (co1::Scaled{C1} * co2::Scaled{C2}) where {C1,C2}
  c = coefficient(co1) * coefficient(co2)
  o = argument(co1) * argument(co2)
  return c * o
end

function (a1::Prod{A} * a2::Scaled{C,A}) where {C,A}
  return coefficient(a2) * (a1 * argument(a2))
end

function (a1::Prod{A} + a2::Scaled{C,A}) where {C,A}
  return one(C) * a1 + Prod{A}() * a2
end

# (Op("X", 1) + Op("X", 2)) + (Op("X", 3) + Op("X", 4))
# (Op("X", 1) * Op("X", 2) + Op("X", 3) * Op("X", 4)) + (Op("X", 5) * Op("X", 6) + Op("X", 7) * Op("X", 8))
(a1::Sum{A} + a2::Sum{A}) where {A} = Applied(sum, (vcat(a1.args[1], a2.args[1]),))
(a1::Sum{A} - a2::Sum{A}) where {A} = a1 + (-a2)

(a1::Prod{A} * a2::Prod{A}) where {A} = Applied(prod, (vcat(only(a1.args), only(a2.args)),))

(os::Sum{Scaled{C,A}} + o::A) where {C,A} = os + one(C) * o
(o::A + os::Sum{Scaled{C,A}}) where {C,A} = one(C) * o + os

# Op("X", 1) + Op("X", 2) + 1.3 * Op("X", 3)
(os::Sum{A} + co::Scaled{C,A}) where {C,A} = one(C) * os + co

# 1.3 * Op("X", 1) + (Op("X", 2) + Op("X", 3))
(co::Scaled{C,A} + os::Sum{A}) where {C,A} = co + one(C) * os

# 1.3 * (Op("X", 1) + Op("X", 2))
(c::Number * os::Sum) = Applied(sum, (c * os.args[1],))

(a1::Applied * a2::Sum) = Applied(sum, (map(a -> a1 * a, only(a2.args)),))
(a1::Sum * a2::Applied) = Applied(sum, (map(a -> a * a2, only(a1.args)),))
(a1::Sum * a2::Sum) = Applied(prod, ([a1, a2],))

function _expand(a1::Sum, a2::Sum)
  return Applied(sum, (vec([a1[i] * a2[j] for i in 1:length(a1), j in 1:length(a2)]),))
end

function expand(a::Prod)
  if length(a) == 1
    return a[1]
  elseif length(a) ≥ 2
    a12 = _expand(a[1], a[2])
    return expand(Applied(prod, (vcat([a12], a[3:end]),)))
  end
end

# (Op("X", 1) + Op("X", 2)) * 1.3
(os::Sum * c::Number) = c * os

# (Op("X", 1) + Op("X", 2)) / 1.3
(os::Sum / c::Number) = inv(c) * os

# Promotions
function (co1::Scaled{C,Prod{A}} + co2::Scaled{C,A}) where {C,A}
  return co1 + coefficient(co2) * Applied(prod, ([argument(co2)],))
end

function (a1::Scaled - a2::Scaled)
  return a1 + (-a2)
end

function (a1::Prod{A} + a2::A) where {A}
  return a1 + Applied(prod, ([a2],))
end

function (a1::Sum{A} + a2::Prod{A}) where {A}
  return Prod{A}() * a1 + a2
end

function (a1::Sum{A} + a2::Sum{Scaled{C,Prod{A}}}) where {C,A}
  return (one(C) * Prod{A}() * a1) + a2
end

function (a1::Prod{A} - a2::A) where {A}
  return a1 + (-a2)
end

function (co1::Sum{Scaled{C,Prod{A}}} + co2::Scaled{C,A}) where {C,A}
  return co1 + coefficient(co2) * Applied(prod, ([argument(co2)],))
end

function (a1::Sum{Scaled{C1,Prod{A}}} - a2::Scaled{C2,A}) where {C1,C2,A}
  return a1 + (-a2)
end

function (a1::Sum{Scaled{C,Prod{A}}} - a2::Prod{A}) where {C,A}
  return a1 + (-a2)
end

function (a1::Sum{Scaled{C1,Prod{A}}} - a2::Scaled{C2,Prod{A}}) where {C1,C2,A}
  return a1 + (-a2)
end

function (a1::Sum{A} + a2::Scaled{C,Prod{A}}) where {C,A}
  return Sum{Scaled{C,Prod{A}}}() + a1 + a2
end

function (a1::Sum{Scaled{C1,Prod{A}}} + a2::Scaled{C2,A}) where {C1,C2,A}
  C = promote_type(C1, C2)
  return one(C) * a1 + one(C) * a2
end

# (::Sum{Scaled{Bool,Prod{Op}}} + ::Scaled{Float64,Prod{Op}})
function (a1::Sum{Scaled{C1,A}} + a2::Scaled{C2,A}) where {C1,C2,A}
  C = promote_type(C1, C2)
  return one(C) * a1 + one(C) * a2
end

# TODO: Is this needed? It seems like:
#
# (a1::Sum{A} + a2::A)
#
# is not being called.
function (a1::Sum{Scaled{C,A}} + a2::Scaled{C,A}) where {C,A}
  return Applied(sum, (vcat(only(a1.args), [a2]),))
end

function (a1::Sum{Scaled{C,Prod{A}}} + a2::Sum{A}) where {C,A}
  a2 = one(C) * a2
  a2 = Prod{A}() * a2
  return a1 + one(C) * Prod{A}() * a2
end

function (a1::Sum{Prod{A}} + a2::A) where {A}
  return a1 + (Prod{A}() * a2)
end

function (a1::Sum{Prod{A}} + a2::Scaled{C,A}) where {C,A}
  return a1 + (Prod{A}() * a2)
end

function (a1::Sum{Scaled{C,Prod{A}}} + a2::A) where {C,A}
  return a1 + one(C) * a2
end
(a1::Sum{Scaled{C,Prod{A}}} - a2::A) where {C,A} = a1 + (-a2)

function (a1::Sum{Scaled{C,Prod{A}}} + a2::Sum{Scaled{C,A}}) where {C,A}
  return a1 + (Prod{A}() * a2)
end

function (o::A + os::Sum{Scaled{C,Prod{A}}}) where {C,A}
  return one(C) * o + os
end

function (a::Sum^n::Int)
  r = a
  for _ in 2:n
    r *= a
  end
  return r
end

function (a::Prod^n::Int)
  r = a
  for _ in 2:n
    r *= a
  end
  return r
end

exp(a::Applied) = Applied(exp, (a,))

const Exp{A} = Applied{typeof(exp),Tuple{A},NamedTuple{(),Tuple{}}}
const Adjoint{A} = Applied{typeof(adjoint),Tuple{A},NamedTuple{(),Tuple{}}}

argument(a::Exp) = a.args[1]

(c::Number * e::Exp) = Applied(*, (c, e))
(e::Exp * c::Number) = c * e
(e1::Exp * e2::Exp) = Applied(prod, ([e1, e2],))
(e1::Applied * e2::Exp) = Applied(prod, ([e1, e2],))
(e1::Exp * e2::Applied) = Applied(prod, ([e1, e2],))

function reverse(a::Prod)
  return Applied(prod, (reverse(only(a.args)),))
end

adjoint(a::Prod) = Applied(prod, (map(adjoint, reverse(only(a.args))),))

#
# Convenient indexing
#

getindex(a::Union{Sum,Prod}, I...) = only(a.args)[I...]
iterate(a::Union{Sum,Prod}, args...) = iterate(only(a.args), args...)
size(a::Union{Sum,Prod}) = size(only(a.args))
length(a::Union{Sum,Prod}) = length(only(a.args))
firstindex(a::Union{Sum,Prod}) = 1
lastindex(a::Union{Sum,Prod}) = length(a)
keys(a::Union{Sum,Prod}) = 1:length(a)

length(a::Scaled{C,<:Sum}) where {C} = length(argument(a))
length(a::Scaled{C,<:Prod}) where {C} = length(argument(a))
getindex(a::Scaled{C,<:Sum}, I...) where {C} = getindex(argument(a), I...)
getindex(a::Scaled{C,<:Prod}, I...) where {C} = getindex(argument(a), I...)
lastindex(a::Scaled{C,<:Sum}) where {C} = lastindex(argument(a))
lastindex(a::Scaled{C,<:Prod}) where {C} = lastindex(argument(a))

#
# Functions convenient for AutoMPO code
#

terms(a::Union{Sum,Prod}) = only(a.args)
terms(a::Scaled{C,<:Union{Sum,Prod}}) where {C} = terms(argument(a))
copy(a::Applied) = Applied(deepcopy(a.f), deepcopy(a.args), deepcopy(a.kwargs))
Sum(a::Vector) = Applied(sum, (a,))
Prod(a::Vector) = Applied(prod, (a,))
function isless(a1::Applied{F}, a2::Applied{F}) where {F}
  return (isless(a1.args, a2.args) && isless(a1.kwargs, a2.kwargs))
end

#
# Printing
#

function show(io::IO, ::MIME"text/plain", a::Sum)
  print(io, "sum(\n")
  for n in eachindex(a)
    print(io, "  ", a[n])
    if n ≠ lastindex(a)
      print(io, "\n")
    end
  end
  print(io, "\n)")
  return nothing
end
show(io::IO, a::Sum) = show(io, MIME("text/plain"), a)

function show(io::IO, ::MIME"text/plain", a::Prod)
  print(io, "prod(\n")
  for n in eachindex(a)
    print(io, "  ", a[n])
    if n ≠ lastindex(a)
      print(io, "\n")
    end
  end
  print(io, "\n)")
  return nothing
end
show(io::IO, a::Prod) = show(io, MIME("text/plain"), a)

function show(io::IO, m::MIME"text/plain", a::Exp)
  print(io, a.f, "(")
  for n in 1:length(a.args)
    print(io, a.args[n])
    if n < length(a.args)
      print(io, ", ")
    end
  end
  print(io, ")")
  return nothing
end
show(io::IO, a::Exp) = show(io, MIME("text/plain"), a)

function show(io::IO, m::MIME"text/plain", a::Applied)
  print(io, a.f, "(")
  for n in eachindex(a.args)
    print(io, a.args[n])
    if n < length(a.args)
      print(io, ", ")
    end
  end
  if !isempty(a.kwargs)
    print(io, "; ")
    for n in 1:length(a.kwargs)
      print(io, keys(a.kwargs)[n], "=", a.kwargs[n])
      if n < length(a.kwargs)
        print(io, ", ")
      end
    end
  end
  print(io, ")")
  return nothing
end
show(io::IO, a::Applied) = show(io, MIME("text/plain"), a)

end
