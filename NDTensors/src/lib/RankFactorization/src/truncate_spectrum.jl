using ..Vendored.TypeParameterAccessors: unwrap_array_type

## TODO write Exposed version of truncate
function truncate!!(P::AbstractArray; kwargs...)
    return truncate!!(unwrap_array_type(P), P; kwargs...)
end

# CPU version.
function truncate!!(::Type{<:Array}, P::AbstractArray; kwargs...)
    truncerr, docut = truncate!(P; kwargs...)
    return P, truncerr, docut
end

# GPU fallback version, convert to CPU.
function truncate!!(::Type{<:AbstractArray}, P::AbstractArray; kwargs...)
    P_cpu = cpu(P)
    truncerr, docut = truncate!(P_cpu; kwargs...)
    P = adapt(unwrap_array_type(P), P_cpu)
    return P, truncerr, docut
end

# CPU implementation.
function truncate!(
        P::AbstractVector;
        mindim = nothing,
        maxdim = nothing,
        cutoff = nothing,
        use_absolute_cutoff = nothing,
        use_relative_cutoff = nothing,
    )
    mindim = replace_nothing(mindim, default_mindim(P))
    maxdim = replace_nothing(maxdim, length(P))
    cutoff = replace_nothing(cutoff, typemin(eltype(P)))
    use_absolute_cutoff = replace_nothing(use_absolute_cutoff, default_use_absolute_cutoff(P))
    use_relative_cutoff = replace_nothing(use_relative_cutoff, default_use_relative_cutoff(P))

    origm = length(P)
    docut = zero(eltype(P))

    #if P[1] <= 0.0
    #  P[1] = 0.0
    #  resize!(P, 1)
    #  return 0.0, 0.0
    #end

    if origm == 1
        docut = abs(P[1]) / 2
        return zero(eltype(P)), docut
    end

    s = sign(P[1])
    s < 0 && (P .*= s)

    #Zero out any negative weight
    for n in origm:-1:1
        (P[n] >= zero(eltype(P))) && break
        P[n] = zero(eltype(P))
    end

    n = origm
    truncerr = zero(eltype(P))
    while n > maxdim
        truncerr += P[n]
        n -= 1
    end

    if use_absolute_cutoff
        #Test if individual prob. weights fall below cutoff
        #rather than using *sum* of discarded weights
        while P[n] <= cutoff && n > mindim
            truncerr += P[n]
            n -= 1
        end
    else
        scale = one(eltype(P))
        if use_relative_cutoff
            scale = sum(P)
            (scale == zero(eltype(P))) && (scale = one(eltype(P)))
        end

        #Continue truncating until *sum* of discarded probability
        #weight reaches cutoff reached (or m==mindim)
        while (truncerr + P[n] <= cutoff * scale) && (n > mindim)
            truncerr += P[n]
            n -= 1
        end

        truncerr /= scale
    end

    if n < 1
        n = 1
    end

    if n < origm
        docut = (P[n] + P[n + 1]) / 2
        if abs(P[n] - P[n + 1]) < eltype(P)(1.0e-3) * P[n]
            docut += eltype(P)(1.0e-3) * P[n]
        end
    end

    s < 0 && (P .*= s)
    resize!(P, n)
    return truncerr, docut
end
