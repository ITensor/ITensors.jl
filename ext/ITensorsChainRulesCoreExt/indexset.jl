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

rrule(::typeof(adjoint), x::ITensor) = rrule(prime, x)

@non_differentiable permute(::Indices, ::Indices)
