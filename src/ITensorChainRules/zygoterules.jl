
# Needed for defining the rule for `adjoint(A::ITensor)`
# which currently doesn't work by overloading `ChainRulesCore.rrule`
using ZygoteRules: @adjoint

@adjoint function Base.adjoint(x::ITensor)
  y = prime(x)
  function adjoint_pullback(ȳ)
    x̄ = inv_op(prime, ȳ)
    return (x̄,)
  end
  return y, adjoint_pullback
end
