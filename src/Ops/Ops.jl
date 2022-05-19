module Ops

using ..LazyApply

import Base: +, -, *, /, exp, show, adjoint, isless

export Op, OpSum, which_op, site, sites, params, Applied, expand

#####################################################################################
# General functionality
#

# Helper function to split a `Tuple` according to the function `f`.
# For example:
#
# julia> t = (1, "X", 1, 2, "Y", 2, "Z", 4)
# (1, "X", 1, 2, "Y", 2, "Z", 4)
# 
# julia> split(x -> x isa AbstractString, t)
# ((1,), ("X", 1, 2), ("Y", 2), ("Z", 4))
# 
function split(f, t::Tuple)
  n = findall(f, t)
  ti = t[1:(first(n) - 1)]
  ts = ntuple(i -> t[n[i]:(n[i + 1] - 1)], length(n) - 1)
  tf = t[last(n):end]
  return ti, ts..., tf
end

struct Op
  which_op
  site::Tuple
  params::NamedTuple
  function Op(which_op, site::Tuple; kwargs...)
    return new(which_op, site, NamedTuple(kwargs))
  end
end
Op(which_op, site...; kwargs...) = Op(which_op, site; kwargs...)

function isless(o1::Op, o2::Op)
  if site(o1) ≠ site(o2)
    return site(o1) < site(o2)
  end
  if which_op(o1) ≠ which_op(o2)
    return which_op(o1) < which_op(o2)
  end
  return params(s1) < params(s2)
end

function isless(o1::Prod{Op}, o2::Prod{Op})
  if length(o1) ≠ length(o2)
    return length(o1) < length(o2)
  end
  for n in 1:length(o1)
    if o1[n] ≠ o2[n]
      return (o1[n] < o2[n])
    end
  end
  return false
end

function isless(o1::Scaled{C1,Prod{Op}}, o2::Scaled{C2,Prod{Op}}) where {C1,C2}
  if argument(o1) == argument(o2)
    if coefficient(o1) ≈ coefficient(o2)
      return false
    else
      c1 = coefficient(o1)
      c2 = coefficient(o2)
      #"lexicographic" ordering on  complex numbers
      return real(c1) < real(c2) || (real(c1) ≈ real(c2) && imag(c1) < imag(c2))
    end
  end
  return argument(o1) < argument(o2)
end

function Op(t::Tuple)
  which_op = first(t)
  site_params = Base.tail(t)
  if last(site_params) isa NamedTuple
    site = Base.front(site_params)
    params = last(site_params)
  else
    site = site_params
    params = (;)
  end
  return Op(which_op, site; params...)
end

which_op(o::Op) = o.which_op
site(o::Op) = o.site
params(o::Op) = o.params

function sites(a::Union{Sum,Prod})
  s = []
  for n in 1:length(a)
    s = s ∪ site(a[n])
  end
  return sort(map(identity, s))
end
sites(a::Scaled{C,<:Sum}) where {C} = sites(argument(a))
sites(a::Scaled{C,<:Prod}) where {C} = sites(argument(a))

params(a::Scaled{C,<:Prod}) where {C} = params(only(argument(a)))

which_op(a::Scaled{C,Op}) where {C} = which_op(argument(a))
sites(a::Scaled{C,Op}) where {C} = sites(argument(a))
params(a::Scaled{C,Op}) where {C} = params(argument(a))

#
# Op algebra
#

const OpSum{C} = Sum{Scaled{C,Prod{Op}}}

OpSum() = OpSum{Float64}()

(o1::Op + o2::Op) = Applied(sum, ([o1, o2],))
(o1::Op * o2::Op) = Applied(prod, ([o1, o2],))
-(o::Op) = -one(Int) * o
(o1::Op - o2::Op) = o1 + (-o2)

(c::Number * o::Op) = Applied(*, (c, o))
(o::Op * c::Number) = Applied(*, (c, o))
(o::Op / c::Number) = Applied(*, (inv(c), o))

(c::Number * o::Prod{Op}) = Applied(*, (c, o))
(o::Prod{Op} * c::Number) = Applied(*, (c, o))
(o::Prod{Op} / c::Number) = Applied(*, (inv(c), o))

# 1.3 * Op("X", 1) + Op("X", 2)
# 1.3 * Op("X", 1) * Op("X", 2) + Op("X", 3)
(co1::Scaled{C} + o2::Op) where {C} = co1 + one(C) * o2

# Op("X", 1) + 1.3 * Op("X", 2)
(o1::Op + co2::Scaled{C}) where {C} = one(C) * o1 + co2

# 1.3 * Op("X", 1) + Op("X", 2) * Op("X", 3)
# 1.3 * Op("X", 1) * Op("X", 2) + Op("X", 3) * Op("X", 4)
(co1::Scaled{C} + o2::Prod{Op}) where {C} = co1 + one(C) * o2

# 1.3 * Op("X", 1) * Op("X", 2)
(co1::Scaled{C} * o2::Op) where {C} = co1 * (one(C) * o2)

exp(o::Op) = Applied(exp, (o,))

adjoint(o::Op) = Applied(adjoint, (o,))
adjoint(o::LazyApply.Adjoint{Op}) = only(o.args)

(o1::Exp{Op} * o2::Op) = Applied(prod, ([o1, o2],))

#
# Tuple interface
#

const OpSumLike{C} = Union{Sum{Op},Sum{Scaled{C,Op}},Sum{Prod{Op}},Sum{Scaled{C,Prod{Op}}}}

# Make a `Scaled{C,Prod{Op}}` from a `Tuple` input,
# for example:
#
# (1.2, "X", 1, "Y", 2) -> 1.2 * Op("X", 1) * Op("Y", 2)
#
function op_term(a::Tuple{Number,Vararg})
  c = first(a)
  return c * op_term(Base.tail(a))
end

function op_term(a::Tuple{Vararg})
  a_split = split(x -> x isa AbstractString, a)
  @assert isempty(first(a_split))
  a_split = Base.tail(a_split)
  o = Op(first(a_split))
  for aₙ in Base.tail(a_split)
    o *= Op(aₙ)
  end
  return o
end

function (a1::OpSumLike + a2::Tuple)
  return a1 + op_term(a2)
end

function (a1::OpSumLike - a2::Tuple{Number,Vararg})
  return a1 + (-first(a2), Base.tail(a2)...)
end

function (a1::OpSumLike - a2::Tuple{AbstractString,Vararg})
  return a1 + (-true, a2...)
end

function (a1::Prod{Op} * a2::Tuple)
  return a1 * op_term(a2)
end

function (a1::Scaled{C,Prod{Op}} * a2::Tuple) where {C}
  return a1 * op_term(a2)
end

function show(io::IO, ::MIME"text/plain", o::Op)
  print(io, which_op(o))
  print(io, site(o))
  if !isempty(params(o))
    print(io, params(o))
  end
  return nothing
end
show(io::IO, o::Op) = show(io, MIME("text/plain"), o)

function show(io::IO, ::MIME"text/plain", o::Prod{Op})
  for oₙ in o
    print(io, oₙ, " ")
  end
  return nothing
end
show(io::IO, o::Prod{Op}) where {C} = show(io, MIME("text/plain"), o)

function show(io::IO, m::MIME"text/plain", o::Scaled{C,Prod{Op}}) where {C}
  print(io, coefficient(o))
  print(io, " ")
  show(io, m, argument(o))
  return nothing
end
show(io::IO, o::Scaled{C,Prod{Op}}) where {C} = show(io, MIME("text/plain"), o)

function show(io::IO, ::MIME"text/plain", o::LazyApply.Adjoint{Op})
  print(io, o')
  print(io, "'")
  return nothing
end
show(io::IO, o::LazyApply.Adjoint{Op}) = show(io, MIME("text/plain"), o)

end
