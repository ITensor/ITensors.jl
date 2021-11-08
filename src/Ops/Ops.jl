module Ops

using Compat
using LinearAlgebra
using Zeros
using ..LazyApply

using ..LazyApply: ∑, ∏, α, coefficient, Applied

import ..LazyApply: coefficient

import Base: show, *, /, +, -, Tuple, one, exp, adjoint, promote_rule, convert

export Op, sites, params

# TODO: Add this once merged with ITensors.jl.
#export OpSum

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

#
# General functionality
#####################################################################################

#####################################################################################
# LazyApply extensions
# TODO: Move to `LazyApply`
#

# Helper function for determing the cofficient type of an `Op` related type.
coefficient_type(o::Type) = One
coefficient_type(o::Type{<:α{<:Any,T}}) where {T} = T
coefficient_type(o::Type{<:∑{T}}) where {T} = coefficient_type(T)

coefficient_type(o::Applied) = coefficient_type(typeof(o))

#
# LazyApply extensions
#####################################################################################

const WhichOp = Union{String,AbstractMatrix,UniformScaling}

struct Op
  which_op::WhichOp
  sites::Tuple{Vararg{Int}}
  params::NamedTuple
end
which_op(o::Op) = o.which_op
sites(o::Op) = o.sites
params(o::Op) = o.params
op(o::Op) = o
coefficient_type(o::Op) = One
coefficient(o::Op) = one(coefficient_type(o))

params(o::Vector{Op}) = params(only(o))

# exp
exp(o::Op) = Applied(exp, o)

# adjoint
adjoint(o::Op) = Applied(adjoint, o)

Tuple(o::Op) = (which_op(o), sites(o), params(o))

const OpExpr = Union{Op,∏{Op},∑{Op},∑{∏{Op}},α{Op},α{∏{Op}},∑{<:α{∏{Op}}}}

# Using `OpSum` for external interface.
const OpSum{T} = ∑{α{∏{Op},T}}

# Type promotion and conversion
convert(::Type{α{Op,T}}, o::Op) where {T} = one(T) * o
convert(::Type{∏{Op}}, o::Op) = ∏([o])
convert(::Type{∑{Op}}, o::Op) = ∑([o])
convert(::Type{α{∏{Op},T}}, o::Op) where {T} = one(T) * convert(∏{Op}, o)
function convert(::Type{∑{α{∏{Op},T}}}, o::Op) where {T}
  return ∑([convert(α{∏{Op},T}, o)])
end

convert(O::Type{<:Op}, o::Tuple) = O(o)
convert(O::Type{<:α{Op}}, o::Tuple) = convert(O, Op(o))
convert(O::Type{<:∏{Op}}, o::Tuple) = convert(O, Op(o))
convert(O::Type{<:∑{Op}}, o::Tuple) = convert(O, Op(o))
convert(O::Type{<:α{∏{Op}}}, o::Tuple) = convert(O, Op(o))
convert(O::Type{<:∑{α{∏{Op},T}} where {T}}, o::Tuple) = convert(O, Op(o))
convert(O::Type{∑{α{∏{Op},T}} where {T}}, o::Tuple) = convert(O, Op(o))

convert(::Type{∑{<:α{∏{Op}}}}, o) = convert(∑{α{∏{Op},T}} where {T}, o)
∑{<:α{∏{Op}}}(o) = (∑{α{∏{Op},T}} where {T})(o)

function (∑{α{∏{Op},T}} where {T})(o::OpExpr)
  return convert(∑{α{∏{Op},T}} where {T}, o)
end
function (∑{α{∏{Op},T}} where {T})(o::Tuple)
  return convert(∑{α{∏{Op},T}} where {T}, o)
end
function (∑{α{∏{Op},T}} where {T})(o::Vector{<:Union{OpExpr,Tuple}})
  return reduce(+, o; init=(∑{α{∏{Op},T}} where {T})())
end
function (∑{α{∏{Op},T}} where {T})(o::WhichOp, args...)
  return convert(∑{α{∏{Op},T}} where {T}, Op(o, args...))
end
function (∑{α{∏{Op},T}} where {T})(c::Number, o::WhichOp, args...)
  return convert(∑{α{∏{Op},T}} where {T}, Op(c, o, args...))
end

# Default constructors
(∑{α{∏{S},T}} where {T})() where {S} = ∑{α{∏{S},Zero}}()
(∑{α{∏{Op},T}} where {T})(o) = convert(∑{α{∏{Op},T}} where {T}, o)

function convert(O::Type{α{∏{Op},T}}, o::α{Op}) where {T}
  return convert(T, coefficient(o)) * ∏([op(o)])
end
function convert(O::Type{∑{α{∏{Op},T}}}, o::α{Op}) where {T}
  return ∑([convert(α{∏{Op},T}, o)])
end

convert(O::Type{∑{∏{Op}}}, o::∏{Op}) = ∑([o])
convert(O::Type{α{∏{Op},T}}, o::∏{Op}) where {T} = one(T) * o
function convert(O::Type{∑{α{∏{Op},T}}}, o::∏{Op}) where {T}
  return ∑([convert(α{∏{Op},T}, o)])
end
convert(O::Type{∑{α{∏{Op},T}}}, o::∑{∏{Op}}) where {T} = one(T) * o

function convert(O::Type{∑{α{∏{Op},T}}}, o::α{∏{Op}}) where {T}
  return ∑([convert(α{∏{Op},T}, o)])
end

# Versions where the type paramater is left out.
function convert(O::Type{∑{α{∏{Op},T}} where {T}}, o)
  return convert(∑{α{∏{Op},coefficient_type(o)}}, o)
end

#
# Promotion rules.
#
# Rules for promoting Op-like objects when they are being added together.
#
# Should cover promotions between these types:
#
# Op
# α{Op,T}
# ∏{Op}
# ∑{Op}
# ∑{α{Op,T}}
# α{∏{Op},T}}
# ∑{∏{Op}}
# ∑{α{∏{Op},T}}
#

# Conversion of `Op`
promote_rule(::Type{Op}, O::Type{<:α{Op}}) = O
promote_rule(::Type{Op}, O::Type{<:∏{Op}}) = O
promote_rule(::Type{Op}, O::Type{<:∑{Op}}) = O
promote_rule(::Type{Op}, O::Type{<:∑{α{Op}}}) = O
promote_rule(::Type{Op}, O::Type{<:α{∏{Op}}}) = O
promote_rule(::Type{Op}, O::Type{<:∑{∏{Op}}}) = O
promote_rule(::Type{Op}, O::Type{<:∑{α{∏{Op}}}}) = O

# Conversion of `α{Op}`
function promote_rule(::Type{α{Op,T}}, ::Type{α{Op,S}}) where {T,S}
  return α{Op,promote_type(T, S)}
end
promote_rule(::Type{α{Op,T}}, ::Type{∏{Op}}) where {T} = α{∏{Op},T}
promote_rule(::Type{α{Op,T}}, ::Type{∑{Op}}) where {T} = ∑{α{Op,T}}
function promote_rule(::Type{α{Op,T}}, ::Type{∑{α{Op,S}}}) where {T,S}
  return ∑{α{Op,promote_type(T, S)}}
end
function promote_rule(::Type{α{Op,T}}, ::Type{α{∏{Op},S}}) where {T,S}
  return α{∏{Op},promote_type(T, S)}
end
function promote_rule(::Type{α{Op,T}}, ::Type{∑{∏{Op}}}) where {T}
  return ∑{α{∏{Op},T}}
end
function promote_rule(::Type{α{Op,T}}, ::Type{∑{α{∏{Op},S}}}) where {T,S}
  return ∑{α{∏{Op},promote_type(T, S)}}
end

# Conversion of `∏{Op}`
promote_rule(::Type{∏{Op}}, ::Type{∑{Op}}) = ∑{∏{Op}}
function promote_rule(::Type{∏{Op}}, ::Type{∑{α{Op,S}}}) where {S}
  return ∑{α{∏{Op},S}}
end
promote_rule(::Type{∏{Op}}, ::Type{α{∏{Op},S}}) where {S} = α{∏{Op},S}
promote_rule(::Type{∏{Op}}, ::Type{∑{∏{Op}}}) = ∑{∏{Op}}
function promote_rule(::Type{∏{Op}}, ::Type{∑{α{∏{Op},S}}}) where {S}
  return ∑{α{∏{Op},S}}
end

# Conversion of `∑{Op}`
promote_rule(::Type{∑{Op}}, ::Type{∑{α{Op,S}}}) where {S} = ∑{α{Op,S}}
function promote_rule(::Type{∑{Op}}, ::Type{α{∏{Op},S}}) where {S}
  return ∑{α{∏{Op},S}}
end
promote_rule(::Type{∑{Op}}, ::Type{∑{∏{Op}}}) = ∑{∏{Op}}
function promote_rule(::Type{∑{Op}}, ::Type{∑{α{∏{Op},S}}}) where {S}
  return ∑{α{∏{Op},S}}
end

# Conversion of `∑{α{Op,T}}`
function promote_rule(::Type{∑{α{Op,T}}}, ::Type{∑{α{Op,S}}}) where {T,S}
  return ∑{α{Op,promote_type(T, S)}}
end
function promote_rule(::Type{∑{α{Op,T}}}, ::Type{α{∏{Op},S}}) where {T,S}
  return ∑{α{∏{Op},promote_type(T, S)}}
end
function promote_rule(::Type{∑{α{Op,T}}}, ::Type{∑{∏{Op}}}) where {T}
  return ∑{α{∏{Op},T}}
end
function promote_rule(::Type{∑{α{Op,T}}}, ::Type{∑{α{∏{Op},S}}}) where {T,S}
  return ∑{α{∏{Op},promote_type(T, S)}}
end

# Conversion of `α{∏{Op},T}`
function promote_rule(::Type{α{∏{Op},T}}, ::Type{α{∏{Op},S}}) where {T,S}
  return α{∏{Op},promote_type(T, S)}
end
function promote_rule(::Type{α{∏{Op},T}}, ::Type{∑{∏{Op}}}) where {T}
  return ∑{α{∏{Op},T}}
end
function promote_rule(::Type{α{∏{Op},T}}, ::Type{∑{α{∏{Op},S}}}) where {T,S}
  return ∑{α{∏{Op},promote_type(T, S)}}
end

# Conversion of `∑{∏{Op}}`
function promote_rule(::Type{∑{∏{Op}}}, ::Type{∑{α{∏{Op},S}}}) where {S}
  return ∑{α{∏{Op},S}}
end

# Conversion of `∑{α{∏{Op},T}}`
function promote_rule(::Type{∑{α{∏{Op},T}}}, ::Type{∑{α{∏{Op},S}}}) where {T,S}
  return ∑{α{∏{Op},promote_type(T, S)}}
end

op(o::α) = o.args[2]
sites(o::α) = sites(op(o))

which_op(o::α{Op}) = which_op(op(o))
params(o::α{Op}) = params(op(o))
one(o::α{Op}) = one(coefficient(o))

sites(o::Union{∑,∏}) = unique(Iterators.flatten(Iterators.map(sites, o)))

# General definition for single-tensor operations like `exp` or `adjoint`.
# F: exp, adjoint, etc.
op(o::Applied{F}) where {F} = o.args[1]
sites(o::Applied{F}) where {F} = sites(op(o))
which_op(o::Applied{F}) where {F} = which_op(op(o))
params(o::Applied{F}) where {F} = params(op(o))

Op(o::Tuple) = Op(o...)
Op(which_op::WhichOp, sites::Tuple; kwargs...) = Op(which_op, sites, values(kwargs))
Op(which_op::WhichOp, sites::Int...; kwargs...) = Op(which_op, sites; kwargs...)
Op(which_op::WhichOp, sites::Vector{Int}; kwargs...) = Op(which_op, Tuple(sites); kwargs...)
function Op(which_op::WhichOp, sites_params::Union{Int,<:NamedTuple}...)
  return Op(which_op, Base.front(sites_params), last(sites_params))
end
Op(α::Number, which_op::WhichOp, args...; kwargs...) = α * Op(which_op, args...; kwargs...)
function Op(which_op::WhichOp, sites_params::Union{Int,WhichOp,NamedTuple}...)
  ts = split(x -> x isa WhichOp, (which_op, sites_params...))
  args = filter(x -> !(x isa Tuple{}), ts)
  return ∏(collect(Op.(args)))
end

# Lazy operations with Op
(arg1::Number * arg2::Op) = α(arg1, arg2)
(arg1::Op / arg2::Number) = inv(arg2) * arg1
(arg1::Op * arg2::Op) = ∏([arg1, arg2])
(arg1::Op + arg2::Op) = ∑([arg1, arg2])
-(o::Op) = -𝟏 * o

# Rules for adding and subtracting Tuples
(arg1::OpExpr + arg2::Tuple) = arg1 + Op(arg2)
(arg1::Tuple + arg2::OpExpr) = Op(arg1) + arg2
(arg1::OpExpr - arg2::Tuple) = arg1 - Op(arg2)
(arg1::Tuple - arg2::OpExpr) = Op(arg1) - arg2

function print_sites(io::IO, sites)
  nsites = length(sites)
  for n in 1:nsites
    print(io, sites[n])
    if n < nsites
      print(io, ", ")
    end
  end
end

function show(io::IO, ::MIME"text/plain", o::Op)
  print(io, which_op(o), "(")
  print_sites(io, sites(o))
  if !isempty(params(o))
    print(io, ", ", params(o))
  end
  return print(io, ")")
end

function show(io::IO, ::MIME"text/plain", o::∏{Op})
  for n in 1:length(o)
    print(io, o[n])
    if n < length(o)
      print(io, " * ")
    end
  end
end

function print_coefficient(io::IO, o)
  return print(io, o)
end

function print_coefficient(io::IO, o::Complex)
  return print(io, "(", o, ")")
end

function show(io::IO, ::MIME"text/plain", o::Union{α{Op},α{∏{Op}}})
  print_coefficient(io, coefficient(o))
  print(io, " ")
  return print(io, op(o))
end

function show(io::IO, ::MIME"text/plain", o::Union{∑{Op},∑{<:α{Op}},∑{∏{Op}},∑{<:α{∏{Op}}}})
  for n in 1:length(o)
    print(io, o[n])
    if n < length(o)
      print(io, " +\n")
    end
  end
end

show(io::IO, o::OpExpr) = show(io, MIME("text/plain"), o)

end
