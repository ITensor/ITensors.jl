export truncate!

function truncate!(P::Vector{Float64};
                   kwargs...)::Tuple{Float64,Float64}
  maxdim::Int = min(get(kwargs,:maxdim,length(P)), length(P))
  mindim::Int = max(get(kwargs,:mindim,1), 1)
  cutoff::Float64 = max(get(kwargs,:cutoff,0.0), 0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)

  origm = length(P)
  docut = 0.0

  if P[1]<=0.0
    P[1] = 0.0
    resize!(P,1)
    return 0.,0.
  end

  if origm==1
    docut = P[1]/2
    return 0.,docut
  end

  #Zero out any negative weight
  for n=origm:-1:1
    (P[n] >= 0.0) && break
    P[n] = 0.0
  end

  n = origm
  truncerr = 0.0
  while n > maxdim
    truncerr += P[n]
    n -= 1
  end

  if absoluteCutoff
    #Test if individual prob. weights fall below cutoff
    #rather than using *sum* of discarded weights
    while P[n] <= cutoff && n > mindim
      truncerr += P[n]
      n -= 1
    end
  else
    scale = 1.0
    if doRelCutoff
      scale = sum(P)
      (scale==0.0) && (scale = 1.0)
    end

    #Continue truncating until *sum* of discarded probability 
    #weight reaches cutoff reached (or m==mindim)
    while (truncerr+P[n] <= cutoff*scale) && (n > mindim)
      truncerr += P[n]
      n -= 1
    end

    truncerr /= scale
  end

  if n < 1
    n = 1
  end

  if n < origm
    docut = (P[n]+P[n+1])/2
    if abs(P[n]-P[n+1]) < 1E-3*P[n]
      docut += 1E-3*P[n]
    end
  end

  resize!(P,n)

  return truncerr,docut
end

