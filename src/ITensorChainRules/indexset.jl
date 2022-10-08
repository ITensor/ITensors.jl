for fname in (
  :prime,
  :setprime,
  :noprime,
  :replaceprime,
  :swapprime,
  :addtags,
  :removetags,
  :replacetags,
  :settags,
  :swaptags,
  :replaceind,
  :replaceinds,
  :swapind,
  :swapinds,
)
  @eval begin
    function rrule(f::typeof($fname), x::ITensor, a...; kwargs...)
      y = f(x, a...; kwargs...)
      function f_pullback(ȳ)
        x̄ = replaceinds(unthunk(ȳ), inds(y) => inds(x))
        ā = map_notangent(a)
        return (NoTangent(), x̄, ā...)
      end
      return y, f_pullback
    end
  end
end

for fname in (
  :prime, :setprime, :noprime, :replaceprime, :addtags, :removetags, :replacetags, :settags
)
  @eval begin
    function rrule(f::typeof($fname), x::Union{MPS,MPO}, a...; kwargs...)
      y = f(x, a...; kwargs...)
      function f_pullback(ȳ)
        x̄ = copy(unthunk(ȳ))
        for j in eachindex(x̄)
          x̄[j] = replaceinds(ȳ[j], inds(y[j]) => inds(x[j]))
        end
        ā = map_notangent(a)
        return (NoTangent(), x̄, ā...)
      end
      return y, f_pullback
    end
  end
end

rrule(::typeof(adjoint), x::Union{ITensor,MPS,MPO}) = rrule(prime, x)

@non_differentiable permute(::Indices, ::Indices)
