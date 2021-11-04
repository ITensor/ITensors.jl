module Ops

using LinearAlgebra
using Zeros
using ..LazyApply

using ..LazyApply: SumOrProd, ∑, Sum, Add, ∏, Prod, Mul, Scaled, ScaledProd, coefficient, Applied

import ..LazyApply: coefficient, _prod

import Base: show, *, /, +, -, Tuple, one, exp, adjoint

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

struct Op{OpNameOrMatrix,N,Params<:NamedTuple}
  name_or_matrix::OpNameOrMatrix
  sites::NTuple{N,Int}
  params::Params
end
name(o::Op) = o.name_or_matrix
matrix(o::Op) = name(o)
sites(o::Op) = o.sites
params(o::Op) = o.params
op(o::Op) = o
coefficient(o::Op) = 𝟏

# exp
exp(o::Op) = Applied(exp, o)

# adjoint
adjoint(o::Op) = Applied(adjoint, o)

Tuple(o::Op) = (name(o), sites(o), params(o))

const OpProd{O} = Prod{Tuple{Vector{O}}} where {O<:Op}
const SimpleOpSum{O} = Sum{Tuple{Vector{O}}} where {O<:Op}
const ScaledOp{T,O} = Mul{Tuple{T,O}} where {O<:Op}
const ScaledOpProd{T,O} = ScaledProd{T,Tuple{Vector{O}}} where {O<:Op}

# This is actually a `ScaledOpProdSum`, using `OpSum` for simplicity.
const OpSum{T,O} = Sum{Tuple{Vector{ScaledOpProd{T,O}}}} where {O<:Op}

op(o::Scaled) = o.args[2]
sites(o::Scaled) = sites(op(o))

name(o::ScaledOp) = name(op(o))
params(o::ScaledOp) = params(op(o))
one(o::ScaledOp) = one(coefficient(o))

sites(o::SumOrProd) = unique(Iterators.flatten(Iterators.map(sites, o)))

# General definition for single-tensor operations like `exp` or `adjoint`.
# F: exp, adjoint, etc.
op(o::Applied{F}) where {F} = o.args[1]
sites(o::Applied{F}) where {F} = sites(op(o))
name(o::Applied{F}) where {F} = name(op(o))
matrix(o::Applied{F}) where {F} = name(o)
params(o::Applied{F}) where {F} = params(op(o))

const OpName = Union{String,AbstractMatrix,UniformScaling}

Op(o::Tuple) = Op(o...)
Op(name::OpName, sites::Tuple; kwargs...) = Op(name, sites, values(kwargs))
Op(name::OpName, sites::Int...; kwargs...) = Op(name, sites; kwargs...)
Op(name::OpName, sites::Vector{Int}; kwargs...) = Op(name, Tuple(sites); kwargs...)
Op(name::OpName, sites_params::Union{Int,<:NamedTuple}...) = Op(name, Base.front(sites_params), last(sites_params))
Op(α::Number, name::OpName, args...; kwargs...) = α * Op(name, args...; kwargs...)
function Op(name::OpName, sites_params::Union{Int,OpName,NamedTuple}...)
  ts = split(x -> x isa OpName, (name, sites_params...))
  args = filter(x -> !(x isa Tuple{}), ts)
  return ∏(collect(Op.(args)))
end

-(o::Op) = -𝟏 * o

OpSum() = OpSum{One,Op}()
OpSum(arg::Op) = ∑([𝟏 * ∏(Op[arg])])
OpSum(arg::ScaledOp) = ∑([arg.args[1] * ∏(Op[arg.args[2]])])
OpSum(arg::ScaledOpProd{T}) where {T} = ∑(ScaledOpProd{T,Op}[arg])
OpSum(arg::Tuple) = OpSum(Op(arg))
OpSum(args...) = OpSum(args)

# Lazy operations
(arg1::Number * arg2::Op) = Mul(arg1, arg2)
(arg1::Op * arg2::Op) = ∏([arg1, arg2])
(arg1::Op + arg2::Op) = ∑([arg1, arg2])

# Maybe to type promotion.
(arg1::ScaledOp + arg2::ScaledOp) = ∑([arg1, arg2])
(arg1::ScaledOp + arg2::Op) = arg1 + one(arg1) * arg2
(arg1::Op + arg2::ScaledOp) = one(arg2) * arg1 + arg2

# Automate conversions
(arg1::Op + arg2::Tuple) = arg1 + Op(arg2)
(arg1::Tuple + arg2::Op) = Op(arg1) + arg2
(arg1::ScaledOp + arg2::Tuple) = arg1 + Op(arg2)
(arg1::Tuple + arg2::ScaledOp) = Op(arg1) + arg2
(arg1::OpSum + arg2::Tuple) = arg1 + Op(arg2)
(arg1::Tuple + arg2::OpSum) = Op(arg1) + arg2

# Special rule since generic `(::Sum + ::Sum)` fails promotion.
function (arg1::OpSum{T1} + arg2::OpSum{T2}) where {T1,T2}
  T = promote_type(T1, T2)
  return OpSum{T,Op}(vcat(arg1.args..., arg2.args...))
end

# Special rule since generic `(::Sum + ::Any)` fails promotion.
(arg1::OpSum + arg2::Op) = arg1 + OpSum(arg2)
(arg1::Op + arg2::OpSum) = OpSum(arg1) + arg2
(arg1::OpSum + arg2::ScaledOp) = arg1 + OpSum(arg2)
(arg1::ScaledOp + arg2::OpSum) = OpSum(arg1) + arg2
(arg1::OpSum + arg2::ScaledOpProd) = arg1 + OpSum(arg2)
(arg1::ScaledOpProd + arg2::OpSum) = OpSum(arg1) + arg2

function show(io::IO, ::MIME"text/plain", o::Op)
  print(io, "(", name(o))
  for s in sites(o)
    print(io, ", ", s)
  end
  if !isempty(params(o))
    print(io, ", ", params(o))
  end
  print(io, ")")
end

show(io::IO, o::Op) = show(io, MIME("text/plain"), o)

end
