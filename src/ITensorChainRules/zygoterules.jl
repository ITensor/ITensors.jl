using ZygoteRules: @adjoint

# Needed for defining the rule for `adjoint(A::ITensor)`
# which currently doesn't work by overloading `ChainRulesCore.rrule`
@adjoint function Base.adjoint(x::Union{ITensor,MPS,MPO})
  y = prime(x)
  function adjoint_pullback(ȳ)
    x̄ = inv_op(prime, ȳ)
    return (x̄,)
  end
  return y, adjoint_pullback
end
