using ChainRulesCore: ChainRulesCore
using ITensors: ITensor
using ZygoteRules: @adjoint

# Needed for defining the rule for `adjoint(A::ITensor)`
# which currently doesn't work by overloading `ChainRulesCore.rrule`
# since it is defined in `Zygote`, which takes precedent.
@adjoint function Base.adjoint(x::ITensor)
    y, adjoint_rrule_pullback = ChainRulesCore.rrule(adjoint, x)
    adjoint_pullback(ȳ) = Base.tail(adjoint_rrule_pullback(ȳ))
    return y, adjoint_pullback
end
