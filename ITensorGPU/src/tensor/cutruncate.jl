function truncate!(P::CuVector{Float64};
                   kwargs...)::Tuple{Float64,Float64,CuVector{Float64}}
  maxdim::Int = min(get(kwargs,:maxdim,length(P)), length(P))
  mindim::Int = min(get(kwargs,:mindim,1), maxdim)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
  origm = length(P)
  docut = 0.0
  maxP  = maximum(P)
  if maxP == 0.0
    P = CUDA.zeros(Float64, 1)
    return 0.,0.,P
  end
  if origm==1
    docut = maxP/2
    return 0., docut, P[1:1]
  end

  @timeit "setup rP" begin
      #Zero out any negative weight
      #neg_z_f = (!signbit(x) ? x : 0.0)
      rP = map(x -> !signbit(x) ? x : 0.0, P)
      n = origm
      truncerr = 0.0
      if n > maxdim
          truncerr = sum(rP[1:n-maxdim])
          n = maxdim
      end
  end
  @timeit "handle cutoff" begin
      if absoluteCutoff
        #Test if individual prob. weights fall below cutoff
        #rather than using *sum* of discarded weights
        sub_arr = rP .- cutoff
        err_rP  = sub_arr ./ abs.(sub_arr)
        flags   = reinterpret(Float64, (signbit.(err_rP) .<< 1 .& 2) .<< 61)
        cut_ind = CUDA.CUBLAS.iamax(length(err_rP), err_rP .* flags) - 1
        n = min(maxdim, length(P) - cut_ind)
        n = max(n, mindim)
        truncerr += sum(rP[cut_ind+1:end])
      else
        scale = 1.0
        @timeit "find scale" begin 
            if doRelCutoff
              scale = sum(P)
              scale = scale > 0.0 ? scale : 1.0
            end
        end

        #Continue truncating until *sum* of discarded probability 
        #weight reaches cutoff reached (or m==mindim)
        sub_arr = rP .+ truncerr .- cutoff*scale
        err_rP  = sub_arr ./ abs.(sub_arr)
        flags   = reinterpret(Float64, (signbit.(err_rP) .<< 1 .& 2) .<< 61)
        cut_ind = CUDA.CUBLAS.iamax(length(err_rP), err_rP .* flags) - 1
        if cut_ind > 0
            truncerr += sum(rP[cut_ind+1:end])
            n = min(maxdim, length(P) - cut_ind)
            n = max(n, mindim)
            if scale==0.0
              truncerr = 0.0
            else
              truncerr /= scale
            end
        else # all are above cutoff
            truncerr += sum(rP[1:maxdim])
            n = min(maxdim, length(P) - cut_ind)
            n = max(n, mindim)
            if scale==0.0
              truncerr = 0.0
            else
              truncerr /= scale
            end
        end
      end
  end
  if n < 1
    n = 1
  end
  if n < origm
    hP = collect(P)
    docut = (hP[n]+hP[n+1])/2
    if abs(hP[n]-hP[n+1]) < 1E-3*hP[n]
      docut += 1E-3*hP[n]
    end
  end
  @timeit "setup return" begin
      rinds = 1:n
      rrP   = P[rinds]
  end
  return truncerr,docut,rrP
end
