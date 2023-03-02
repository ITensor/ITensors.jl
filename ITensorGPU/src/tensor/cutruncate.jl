import LinearAlgebra: BlasReal

function truncate!(P::CuVector{T}; kwargs...)::Tuple{T,T,CuVector{T}} where {T<:BlasReal}
  maxdim::Int = min(get(kwargs, :maxdim, length(P)), length(P))
  mindim::Int = min(get(kwargs, :mindim, 1), maxdim)
  cutoff::Float64 = get(kwargs, :cutoff, 0.0)
  absoluteCutoff::Bool = get(kwargs, :absoluteCutoff, false)
  doRelCutoff::Bool = get(kwargs, :doRelCutoff, true)
  origm = length(P)
  docut = zero(T)

  maxP = maximum(P)
  if maxP == zero(T)
    P = CUDA.zeros(T, 1)
    return zero(T), zero(T), P
  end
  if origm == 1
    docut = maxP / 2
    return zero(T), docut, P[1:1]
  end
  @timeit "setup rP" begin
    #Zero out any negative weight
    #neg_z_f = (!signbit(x) ? x : 0.0)
    rP = map(x -> !signbit(x) ? Float64(x) : 0.0, P)
    n = origm
  end
  @timeit "handle cutoff" begin
    if absoluteCutoff
      #Test if individual prob. weights fall below cutoff
      #rather than using *sum* of discarded weights
      sub_arr = rP .- Float64(cutoff)
      err_rP = sub_arr ./ abs.(sub_arr)
      flags = reinterpret(Float64, (signbit.(err_rP) .<< 1 .& 2) .<< 61)
      cut_ind = CUDA.CUBLAS.iamax(length(err_rP), err_rP .* flags) - 1
      if cut_ind > 0
        n = min(maxdim, cut_ind)
        n = max(n, mindim)
      else
        n = maxdim
      end
      truncerr = T(sum(rP[(n + 1):end]))
    else
      truncerr = zero(T)
      scale = one(T)
      @timeit "find scale" begin
        if doRelCutoff
          scale = sum(P)
          scale = scale > zero(T) ? scale : one(T)
        end
      end
      #Truncating until *sum* of discarded probability 
      #weight reaches cutoff reached (or m==mindim)
      csum_rp = Float64.(CUDA.reverse(CUDA.cumsum(CUDA.reverse(rP))))
      sub_arr = csum_rp .- Float64(cutoff * scale)
      err_rP = sub_arr ./ abs.(sub_arr)
      flags = reinterpret(Float64, (signbit.(err_rP) .<< 1 .& 2) .<< 61)
      cut_ind = (CUDA.CUBLAS.iamax(length(err_rP), err_rP .* flags) - 1)
      if cut_ind > 0
        n = min(maxdim, cut_ind)
        n = max(n, mindim)
      else
        n = maxdim
      end
      truncerr = sum(rP[(n + 1):end])
      if scale == zero(T)
        truncerr = zero(T)
      else
        truncerr /= scale
      end
    end
  end
  if n < 1
    n = 1
  end
  if n < origm
    hP = collect(P)
    docut = (hP[n] + hP[n + 1]) / 2
    if abs(hP[n] - hP[n + 1]) < 1E-3 * hP[n]
      docut += T(1E-3) * hP[n]
    end
  end
  @timeit "setup return" begin
    rinds = 1:n
    rrP = P[rinds]
  end
  return truncerr, docut, rrP
end
