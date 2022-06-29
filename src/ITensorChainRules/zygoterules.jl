
# Needed for defining the rule for `adjoint(A::ITensor)`
# which currently doesn't work by overloading `ChainRulesCore.rrule`
using ZygoteRules: @adjoint

@adjoint function Base.adjoint(x::Union{ITensor,MPS,MPO})
  y = prime(x)
  function adjoint_pullback(ȳ)
    x̄ = inv_op(prime, ȳ)
    return (x̄,)
  end
  return y, adjoint_pullback
end

## XXX: raise issue about `tr` being too generically
## defined in ChainRules
##
## using Zygote
## 
## # Needed because by default it was calling the generic
## # rrule for `tr` inside ChainRules
## function rrule(::typeof(tr), x::ITensor; kwargs...)
##   y, tr_pullback_zygote = pullback(ITensors._tr, x; kwargs...)
##   tr_pullback(ȳ) = (NoTangent(), tr_pullback_zygote(ȳ)...)
##   return y, tr_pullback
## end
