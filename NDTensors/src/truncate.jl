export truncate!

function truncate!(P::Vector{Float64}; kwargs...)::Tuple{Float64,Float64}
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
  cutoff::Float64 = max(get(kwargs, :cutoff, 0.0), 0.0)
  use_absolute_cutoff::Bool = get(kwargs, :use_absolute_cutoff, use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs, :use_relative_cutoff, use_relative_cutoff)

  origm = length(P)
  docut = 0.0

  #if P[1] <= 0.0
  #  P[1] = 0.0
  #  resize!(P, 1)
  #  return 0.0, 0.0
  #end

  if origm == 1
    docut = abs(P[1]) / 2
    return 0.0, docut
  end

  s = sign(P[1])
  s < 0 && (P .*= s)

  #Zero out any negative weight
  for n in origm:-1:1
    (P[n] >= 0.0) && break
    P[n] = 0.0
  end

  n = origm
  truncerr = 0.0
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
    scale = 1.0
    if use_relative_cutoff
      scale = sum(P)
      (scale == 0.0) && (scale = 1.0)
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
    if abs(P[n] - P[n + 1]) < 1E-3 * P[n]
      docut += 1E-3 * P[n]
    end
  end

  s < 0 && (P .*= s)
  resize!(P, n)

  return truncerr, docut
end
