Adapt.adapt(to, x::Exposed) = adapt_structure(to, x)
Adapt.adapt_structure(to, x::Exposed) = adapt_structure(to, unexpose(x))

# https://github.com/JuliaGPU/Adapt.jl/pull/51
# TODO: Remove once https://github.com/JuliaGPU/Adapt.jl/issues/71 is addressed.
function Adapt.adapt_structure(to, A::Exposed{<:Any, <:Hermitian})
    return Hermitian(adapt(to, parent(unexpose(A))), Symbol(unexpose(A).uplo))
end
