export truncate!

function truncate!(P::AbstractVector{ElT}; kwargs...)::Tuple{ElT,ElT} where {ElT}
  cutoff::Union{Nothing,ElT} = get(kwargs, :cutoff, zero(ElT))
  if isnothing(cutoff)
    cutoff = typemin(ElT)
  end

  # Keyword argument deprecations
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In truncate!, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs, :absoluteCutoff, use_absolute_cutoff)
  end
  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In truncate!, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs, :doRelCutoff, use_relative_cutoff)
  end

  maxdim::Int = min(get(kwargs, :maxdim, length(P)), length(P))
  mindim::Int = max(get(kwargs, :mindim, 1), 1)

  use_absolute_cutoff::Bool = get(kwargs, :use_absolute_cutoff, use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs, :use_relative_cutoff, use_relative_cutoff)

  origm = length(P)
  docut = zero(ElT)

  #if P[1] <= 0.0
  #  P[1] = 0.0
  #  resize!(P, 1)
  #  return 0.0, 0.0
  #end

  if origm == 1
    docut = abs(P[1]) / 2
    return zero(ElT), docut
  end

  s = sign(P[1])
  s < 0 && (P .*= s)

  #Zero out any negative weight
  for n in origm:-1:1
    (P[n] >= zero(ElT)) && break
    P[n] = zero(ElT)
  end

  n = origm
  truncerr = zero(ElT)
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
    scale = one(ElT)
    if use_relative_cutoff
      scale = sum(P)
      (scale == zero(ElT)) && (scale = one(ElT))
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
    if abs(P[n] - P[n + 1]) < ElT(1e-3) * P[n]
      docut += ElT(1e-3) * P[n]
    end
  end

  s < 0 && (P .*= s)
  resize!(P, n)

  return truncerr, docut
end
