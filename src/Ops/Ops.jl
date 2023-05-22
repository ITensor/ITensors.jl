module Ops

using ..LazyApply

import Base: ==, +, -, *, /, convert, exp, show, adjoint, isless, hash

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
# [(1,), ("X", 1, 2), ("Y", 2), ("Z", 4)]
# 
function split(f, t::Tuple)
  n = findall(f, t)
  nsplit = length(n) + 1
  s = Vector{Any}(undef, nsplit)
  s[1] = t[1:(first(n) - 1)]
  for i in 2:(nsplit - 1)
    s[i] = t[n[i - 1]:(n[i] - 1)]
  end
  s[end] = t[last(n):end]
  return s
end

## XXX: Very long compile times:
## https://github.com/JuliaLang/julia/issues/45545
##
## julia> using ITensors
## 
## julia> @time ITensors.Ops.split(x -> x isa String, ("X", 1))
##   7.588123 seconds (2.34 M allocations: 100.919 MiB, 1.71% gc time, 100.00% compilation time)
## ((), ("X", 1))
## 
## julia> @time ITensors.Ops.split(x -> x isa String, ("X", 1))
##   0.042590 seconds (88.59 k allocations: 4.823 MiB, 19.13% gc time, 99.84% compilation time)
## ((), ("X", 1))
##
## function split(f, t::Tuple)
##   n = findall(f, t)
##   ti = t[1:(first(n) - 1)]
##   ts = ntuple(i -> t[n[i]:(n[i + 1] - 1)], length(n) - 1)
##   tf = t[last(n):end]
##   return ti, ts..., tf
## end

struct Op
  which_op
  sites::Tuple
  params::NamedTuple
  function Op(which_op, site...; kwargs...)
    return new(which_op, site, NamedTuple(kwargs))
  end
end

which_op(o::Op) = o.which_op
name(o::Op) = which_op(o)
sites(o::Op) = o.sites
site(o::Op) = only(sites(o))
params(o::Op) = o.params

function (o1::Op == o2::Op)
  return o1.which_op == o2.which_op && o1.sites == o2.sites && o1.params == o2.params
end

function hash(o::Op, h::UInt)
  return hash(which_op(o), hash(sites(o), hash(params(o), hash(:Op, h))))
end

# Version of `isless` defined for matrices
_isless(a, b) = isless(a, b)
_isless(a::AbstractMatrix, b::AbstractMatrix) = isless(hash(a), hash(b))
_isless(a::AbstractString, b::AbstractMatrix) = true
_isless(a::AbstractMatrix, b::AbstractString) = !_isless(b, a)

function isless(o1::Op, o2::Op)
  if sites(o1) ≠ sites(o2)
    return sites(o1) < sites(o2)
  end
  if which_op(o1) ≠ which_op(o2)
    return _isless(which_op(o1), which_op(o2))
  end
  return params(o1) < params(o2)
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

## function Op(t::Tuple)
##   which_op = first(t)
##   site_params = Base.tail(t)
##   if last(site_params) isa NamedTuple
##     site = Base.front(site_params)
##     params = last(site_params)
##   else
##     site = site_params
##     params = (;)
##   end
##   return Op(which_op, site; params...)
## end

## function Op(t::Tuple{WhichOp,NamedTuple,Vararg}) where {WhichOp}
##   params = t[2]
##   which_op = t[1]
##   sites = t[3:end]
##   return Op(which_op, sites...; params...)
## end

function sites(a::Union{Sum,Prod})
  s = []
  for n in 1:length(a)
    s = s ∪ sites(a[n])
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

function convert(::Type{Scaled{C1,Prod{Op}}}, o::Scaled{C2,Prod{Op}}) where {C1,C2}
  c = convert(C1, coefficient(o))
  return c * argument(o)
end

"""
An `OpSum` represents a sum of operator
terms.

Often it is used to create matrix
product operator (`MPO`) approximation
of the sum of the terms in the `OpSum` oject.
Each term is a product of local operators
specified by names such as `"Sz"` or `"N"`,
times an optional coefficient which
can be real or complex.

Which local operator names are available
is determined by the function `op`
associated with the `TagType` defined by
special Index tags, such as `"S=1/2"`, `"S=1"`,
`"Fermion"`, and `"Electron"`.
"""
const OpSum{C} = Sum{Scaled{C,Prod{Op}}}

# This helps with in-place operations
OpSum() = OpSum{ComplexF64}()

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

(o1::Op * o2::Sum) = Applied(sum, (map(a -> o1 * a, only(o2.args)),))
(o1::Sum * o2::Op) = Applied(sum, (map(a -> a * o2, only(o1.args)),))

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

const OpSumLike{C} = Union{
  Sum{Op},
  Sum{Scaled{C,Op}},
  Sum{Prod{Op}},
  Sum{Scaled{C,Prod{Op}}},
  Prod{Op},
  Scaled{C,Prod{Op}},
}

const WhichOp = Union{AbstractString,AbstractMatrix{<:Number}}

# Make a `Scaled{C,Prod{Op}}` from a `Tuple` input,
# for example:
#
# (1.2, "X", 1, "Y", 2) -> 1.2 * Op("X", 1) * Op("Y", 2)
#
function op_term(a::Tuple{Number,Vararg})
  c = first(a)
  return c * op_term(Base.tail(a))
end

function op_site(which_op, params::NamedTuple, sites...)
  return Op(which_op, sites...; params...)
end

function op_site(which_op, sites_params...)
  if last(sites_params) isa NamedTuple
    sites = Base.front(sites_params)
    params = last(sites_params)
    return Op(which_op, sites...; params...)
  end
  return Op(which_op, sites_params...)
end

function op_term(a::Tuple{Vararg})
  a_split = split(x -> x isa WhichOp, a)
  @assert isempty(first(a_split))
  popfirst!(a_split)
  o = op_site(first(a_split)...)
  popfirst!(a_split)
  for aₙ in a_split
    o *= op_site(aₙ...)
  end
  return o
end

function (o1::OpSumLike + o2::Tuple)
  return o1 + op_term(o2)
end

function (o1::Tuple + o2::OpSumLike)
  return op_term(o1) + o2
end

function (o1::OpSumLike - o2::Tuple)
  return o1 - op_term(o2)
end

function (o1::Tuple - o2::OpSumLike)
  return op_term(o1) - o2
end

function (o1::OpSumLike * o2::Tuple)
  return o1 * op_term(o2)
end

function (o1::Tuple * o2::OpSumLike)
  return op_term(o1) * o2
end

function show(io::IO, ::MIME"text/plain", o::Op)
  print(io, which_op(o))
  print(io, sites(o))
  if !isempty(params(o))
    print(io, params(o))
  end
  return nothing
end
show(io::IO, o::Op) = show(io, MIME("text/plain"), o)

function show(io::IO, ::MIME"text/plain", o::Prod{Op})
  for n in 1:length(o)
    print(io, o[n])
    if n < length(o)
      print(io, " ")
    end
  end
  return nothing
end
show(io::IO, o::Prod{Op}) = show(io, MIME("text/plain"), o)

function show(io::IO, m::MIME"text/plain", o::Scaled{C,O}) where {C,O<:Union{Op,Prod{Op}}}
  c = coefficient(o)
  if isreal(c)
    c = real(c)
  end
  print(io, c)
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
