using ChainRulesCore: ChainRulesCore, unthunk
using Compat: Returns
using ITensors:
  addtags,
  noprime,
  prime,
  removetags,
  replaceinds,
  replaceprime,
  replacetags,
  setprime,
  settags
using ITensors.ITensorMPS: MPO, MPS

for fname in (
  :prime, :setprime, :noprime, :replaceprime, :addtags, :removetags, :replacetags, :settags
)
  @eval begin
    function ChainRulesCore.rrule(f::typeof($fname), x::Union{MPS,MPO}, a...; kwargs...)
      y = f(x, a...; kwargs...)
      function f_pullback(ȳ)
        x̄ = copy(unthunk(ȳ))
        for j in eachindex(x̄)
          x̄[j] = replaceinds(ȳ[j], inds(y[j]) => inds(x[j]))
        end
        ā = map(Returns(NoTangent()), a)
        return (NoTangent(), x̄, ā...)
      end
      return y, f_pullback
    end
  end
end

ChainRulesCore.rrule(::typeof(adjoint), x::Union{MPS,MPO}) = ChainRulesCore.rrule(prime, x)
