using ChainRulesCore
using ITensors

using ITensors.NDTensors
using ITensors: MPOTerm, OpTerm, SiteOp, coef, ops, sites, Indices, permute

import ChainRulesCore: rrule

function rrule(::typeof(+), x1::OpSum, x2::MPOTerm)
  y = x1 + x2
  function add_pullback(ȳ)
    c̄ = ȳ.data[1].coef
    ō = ȳ.data[1].ops
    x̄1 = OpSum([MPOTerm(c̄, x2.ops)])
    x̄2 = MPOTerm(c̄, x2.ops)
    return (NoTangent(), x̄1, x̄2)
  end
  function add_pullback(ȳ::Vector{Bool})
    x̄1 = OpSum(MPOTerm[])
    x̄2 = MPOTerm(0.0, SiteOp[])
    return (NoTangent(), x̄1, x̄2)
  end
  return y, add_pullback
end

function rrule(::Type{SiteOp}, x1::String, x2::NTuple{N,Int}, x3::NamedTuple) where {N}
  y = SiteOp(x1, x2, x3)
  function SiteOp_pullback(ȳ)
    x̄1 = NoTangent()
    x̄2 = NoTangent()
    x̄3 = ȳ.params
    return (NoTangent(), x̄1, x̄2, x̄3)
  end
  return y, SiteOp_pullback
end

function rrule(::Type{MPOTerm}, x1::Number, x2::OpTerm)
  y = MPOTerm(x1, x2)
  function MPOTerm_pullback(ȳ)
    x̄1 = convert(typeof(x1), ȳ.coef)
    x̄2 = ȳ.ops
    return (NoTangent(), x̄1, x̄2)
  end
  return y, MPOTerm_pullback
end

function rrule(::Type{OpSum}, x1::Vector{MPOTerm})
  y = OpSum(x1)
  function OpSum_pullback(ȳ)
    x̄1 = ȳ.data
    return (NoTangent(), x̄1)
  end
  function OpSum_pullback(ȳ::ZeroTangent)
    x̄1 = MPOTerm[]
    return (NoTangent(), x̄1)
  end
  return y, OpSum_pullback
end

## function rrule(::typeof(getindex), x1::OpSum, x2::Int)
##   y = getindex(x1, x2)
##   function getindex_pullback(ȳ)
##     x̄1 = OpSum([SiteOp(ȳ.coef, ȳ.ops)])
##     x̄2 = NoTangent()
##     return (NoTangent(), x̄1, x̄2)
##   end
##   return y, getindex_pullback
## end

function rrule(::Type{coef}, x1::MPOTerm)
  y = coef(x1)
  function coef_pullback(ȳ)
    x̄1 = MPOTerm(ȳ, x1.ops)
    return (NoTangent(), x̄1)
  end
  return y, coef_pullback
end

function rrule(::Type{ITensor}, x1::Tensor)
  y = ITensor(x1)
  function ITensor_pullback(ȳ)
    x̄1 = Tensor(ȳ)
    return (NoTangent(), x̄1)
  end
  return y, ITensor_pullback
end

function rrule(::typeof(itensor), x1::Tensor)
  y = itensor(x1)
  function itensor_pullback(ȳ)
    x̄1 = tensor(ȳ)
    return (NoTangent(), x̄1)
  end
  return y, itensor_pullback
end

function rrule(f::Type{<:Tensor}, x1::ITensor)
  y = f(x1)
  function Tensor_pullback(ȳ)
    x̄1 = ITensor(ȳ)
    return (NoTangent(), x̄1)
  end
  return y, Tensor_pullback
end

function rrule(::typeof(tensor), x1::ITensor)
  y = tensor(x1)
  function tensor_pullback(ȳ)
    x̄1 = ITensor(typeof(storage(x1))(ȳ.storage.data), inds(x1))
    return (NoTangent(), x̄1)
  end
  return y, tensor_pullback
end

function rrule(::typeof(tensor), x1::TensorStorage, x2::Tuple)
  y = tensor(x1, x2)
  function tensor_pullback(ȳ)
    x̄1 = storage(ȳ)
    x̄2 = NoTangent()
    return (NoTangent(), x̄1, x̄2)
  end
  return y, tensor_pullback
end

function rrule(f::Type{<:Tensor}, x1::TensorStorage, x2::Tuple)
  y = f(x1, x2)
  function tensor_pullback(ȳ)
    x̄1 = copy(storage(x1))
    x̄2 = NoTangent()
    return (NoTangent(), x̄1, x̄2)
  end
  return y, tensor_pullback
end

function rrule(f::Type{<:Dense}, x1::AbstractVector)
  y = f(x1)
  function Dense_pullback(ȳ)
    x̄1 = ȳ.data
    return (NoTangent(), x̄1)
  end
  return y, Dense_pullback
end

## # x1 = exp(a * b), x̄1 = ā * exp(a * b)
## function rrule(::typeof(op), x1::Exp{MPOTerm}, x2::Vector{<:Index})
##   a = x1.args[1].coef
##   b = x1.args[1].ops
##   o = op(b, x2)
##   y = exp(a * o)
##   function op_pullback(ȳ)
##     @show y
##     @show ȳ
## 
##     error("No implemented")
## 
##     x̄2 = NoTangent()
##     return (NoTangent(), x̄1, x̄2)
##   end
##   return y, op_pullback
## end

@non_differentiable sites(::SiteOp)
@non_differentiable support(::Vector{<:SiteOp})
@non_differentiable op(o::Vector{<:SiteOp}, s::Vector{<:Index})
@non_differentiable has_fermion_string(::AbstractString, ::Index)
@non_differentiable permute(::Indices, ::Indices)
@non_differentiable combiner(::Indices)
