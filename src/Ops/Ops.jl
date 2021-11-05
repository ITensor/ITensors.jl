module Ops

using LinearAlgebra
using Zeros
using ..LazyApply

using ..LazyApply:
  SumOrProd, ∑, Sum, Add, ∏, Prod, Mul, Scaled, ScaledProd, coefficient, Applied

import ..LazyApply: coefficient, _prod

import Base: show, *, /, +, -, Tuple, one, exp, adjoint, promote_rule, convert

export Op, sites, params

# TODO: Add this once merged with ITensors.jl.
#export OpSum

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
coefficient(o::Op) = 𝟏

params(o::Vector{Op}) = params(only(o))

# exp
exp(o::Op) = Applied(exp, o)

# adjoint
adjoint(o::Op) = Applied(adjoint, o)

Tuple(o::Op) = (which_op(o), sites(o), params(o))

const ScaledOp{T} = Mul{Tuple{T,Op}}
const ProdOp = Prod{Tuple{Vector{Op}}}
const SumOp = Sum{Tuple{Vector{Op}}}
const SumScaledOp{T} = Sum{Tuple{Vector{ScaledOp{T}}}}
const ScaledProdOp{T} = ScaledProd{T,Tuple{Vector{Op}}}
const SumProdOp = Sum{Tuple{Vector{ProdOp}}}
const SumScaledProdOp{T} = Sum{Tuple{Vector{ScaledProdOp{T}}}}

const OpExpr = Union{Op,ScaledOp,ProdOp,SumOp,SumProdOp,ScaledProdOp,SumScaledProdOp}

# Using `OpSum` for external interface.
const OpSum{T} = SumScaledProdOp{T}

# Default constructors
ScaledOp() = ScaledOp{One}()
ScaledProdOp() = ScaledProdOp{One}()
SumScaledProdOp() = SumScaledProdOp{One}()

# Helper function for determing the cofficient type of an `Op` related type.
coefficient_type(o) = One
coefficient_type(o::ScaledOp{T}) where {T} = T
coefficient_type(o::ScaledProdOp{T}) where {T} = T
coefficient_type(o::SumScaledOp{T}) where {T} = T
coefficient_type(o::SumScaledProdOp{T}) where {T} = T

# Type promotion and conversion
convert(::Type{ScaledOp{T}}, o::Op) where {T} = one(T) * o
convert(::Type{ProdOp}, o::Op) = ∏([o])
convert(::Type{SumOp}, o::Op) = ∑([o])
convert(::Type{ScaledProdOp{T}}, o::Op) where {T} = one(T) * convert(ProdOp, o)
convert(::Type{SumScaledProdOp{T}}, o::Op) where {T} = ∑([convert(ScaledProdOp{T}, o)])

convert(O::Type{<:Op}, o::Tuple) = O(o)
convert(O::Type{<:ScaledOp}, o::Tuple) = convert(O, Op(o))
convert(O::Type{<:ProdOp}, o::Tuple) = convert(O, Op(o))
convert(O::Type{<:SumOp}, o::Tuple) = convert(O, Op(o))
convert(O::Type{<:ScaledProdOp}, o::Tuple) = convert(O, Op(o))
convert(O::Type{<:SumScaledProdOp}, o::Tuple) = convert(O, Op(o))

function convert(O::Type{ScaledProdOp{T}}, o::ScaledOp) where {T}
  return convert(T, coefficient(o)) * ∏([op(o)])
end
function convert(O::Type{SumScaledProdOp{T}}, o::ScaledOp) where {T}
  return ∑([convert(ScaledProdOp{T}, o)])
end

convert(O::Type{ScaledProdOp{T}}, o::ProdOp) where {T} = one(T) * o
convert(O::Type{SumScaledProdOp{T}}, o::ProdOp) where {T} = ∑([convert(ScaledProdOp{T}, o)])

convert(O::Type{SumScaledProdOp{T}}, o::SumProdOp) where {T} = one(T) * o

function convert(O::Type{SumScaledProdOp{T}}, o::ScaledProdOp) where {T}
  return ∑([convert(ScaledProdOp{T}, o)])
end

# Versions where the type paramater is left out.
convert(O::Type{SumScaledProdOp}, o) = convert(SumScaledProdOp{coefficient_type(o)}, o)

promote_rule(::Type{Op}, O::Type{<:ScaledOp}) = O
promote_rule(::Type{Op}, O::Type{<:ProdOp}) = O
promote_rule(::Type{Op}, O::Type{<:SumOp}) = O
promote_rule(::Type{Op}, O::Type{<:ScaledProdOp}) = O
promote_rule(::Type{Op}, O::Type{<:SumScaledProdOp}) = O

function promote_rule(::Type{ScaledOp{T}}, ::Type{ScaledOp{S}}) where {T,S}
  return ScaledOp{promote_type(T, S)}
end
promote_rule(::Type{ScaledOp{T}}, ::Type{ProdOp}) where {T} = ScaledProdOp{T}
function promote_rule(::Type{ScaledOp{T}}, ::Type{ScaledProdOp{S}}) where {T,S}
  return ScaledProdOp{promote_type(T, S)}
end
function promote_rule(::Type{ScaledOp{T}}, ::Type{SumScaledProdOp{S}}) where {T,S}
  return SumScaledProdOp{promote_type(T, S)}
end

function promote_rule(::Type{ScaledProdOp{T}}, ::Type{SumScaledProdOp{S}}) where {T,S}
  return SumScaledProdOp{promote_type(T, S)}
end

op(o::Scaled) = o.args[2]
sites(o::Scaled) = sites(op(o))

which_op(o::ScaledOp) = which_op(op(o))
params(o::ScaledOp) = params(op(o))
one(o::ScaledOp) = one(coefficient(o))

sites(o::SumOrProd) = unique(Iterators.flatten(Iterators.map(sites, o)))

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
(arg1::Number * arg2::Op) = Mul(arg1, arg2)
(arg1::Op / arg2::Number) = inv(arg2) * arg1
(arg1::Op * arg2::Op) = ∏([arg1, arg2])
(arg1::Op + arg2::Op) = ∑([arg1, arg2])
-(o::Op) = -𝟏 * o

# Rules for adding and subtracting Tuples
(arg1::OpExpr + arg2::Tuple) = arg1 + Op(arg2)
(arg1::Tuple + arg2::OpExpr) = Op(arg1) + arg2
(arg1::OpExpr - arg2::Tuple) = arg1 - Op(arg2)
(arg1::Tuple - arg2::OpExpr) = Op(arg1) - arg2

function show(io::IO, ::MIME"text/plain", o::Op)
  print(io, "(", which_op(o))
  for s in sites(o)
    print(io, ", ", s)
  end
  if !isempty(params(o))
    print(io, ", ", params(o))
  end
  return print(io, ")")
end

function show(io::IO, ::MIME"text/plain", o::ProdOp)
  for n in 1:length(o)
    print(io, o[n])
    if n < length(o)
      print(io, " * ")
    end
  end
end

function show(io::IO, ::MIME"text/plain", o::Union{ScaledOp,ScaledProdOp})
  print(io, "(", coefficient(o), ")")
  print(io, " * ")
  return print(io, op(o))
end

function show(io::IO, ::MIME"text/plain", o::Union{SumOp,SumProdOp,SumScaledProdOp})
  for n in 1:length(o)
    print(io, o[n])
    if n < length(o)
      print(io, " +\n")
    end
  end
end

show(io::IO, o::OpExpr) = show(io, MIME("text/plain"), o)

end
