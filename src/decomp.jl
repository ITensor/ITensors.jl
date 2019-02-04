
function truncate!(P::Vector{Float64};
                   kwargs...
                  )::Tuple{Float64,Float64}
  maxm::Int = get(kwargs,:maxm,length(P))
  minm::Int = get(kwargs,:minm,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)

  origm = length(P)
  docut = 0.0

  if P[1]==0.0
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
  while n > maxm
    truncerr += P[n]
    n -= 1
  end

  if absoluteCutoff
    #Test if individual prob. weights fall below cutoff
    #rather than using *sum* of discarded weights
    while P[n] <= cutoff && n > minm
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
    #weight reaches cutoff reached (or m==minm)
    while (truncerr+P[n] <= cutoff*scale) && (n > minm)
      truncerr += P[n]
      n -= 1
    end

    if scale==0.0
      truncerr = 0.0
    else
      truncerr /= scale
    end
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

function factorize(A::ITensor,
                   left_inds::Index...;
                   factorization=factorization)
  Lis = IndexSet(left_inds...)
  #TODO: make this a debug level check
  Lis⊈inds(A) && throw(ErrorException("Input indices must be contained in the ITensor"))

  Ris = difference(inds(A),Lis)
  #TODO: check if A is already ordered properly
  #and avoid doing this permute, since it makes a copy
  #AND/OR use svd!() to overwrite the data of A to save memory
  A = permute(A,Lis...,Ris...)
  if factorization==:QR
    Qis,Qstore,Pis,Pstore = storage_qr(store(A),Lis,Ris)
  elseif factorization==:polar
    Qis,Qstore,Pis,Pstore = storage_polar(store(A),Lis,Ris)
  else
    error("Factorization $factorization not supported")
  end
  return ITensor(Qis,Qstore),ITensor(Pis,Pstore)
end

qr(A::ITensor,left_inds::Index...) = factorize(A,left_inds...;factorization=:QR)

polar(A::ITensor,left_inds::Index...) = factorize(A,left_inds...;factorization=:polar)

"""
    svd(A::ITensor,
        leftind1::Index,
        leftind2::Index,
        ...
        ;kwargs...)

Singular value decomposition (SVD) of an ITensor A, computed
by treating the "left indices" provided collectively
as a row index, and the remaining "right indices" as a
column index (matricization of a tensor).

Whether the SVD performs a trunction depends on the keyword
arguments provided. The following keyword arguments are recognized:
* maxm [Int]
* minm [Int]
* cutoff [Float64]
* truncate [Bool]
"""
function svd(A::ITensor,
             left_inds::Index...;
             kwargs...
            )
  Lis = IndexSet(left_inds...)
  #TODO: make this a debug level check
  Lis⊈inds(A) && throw(ErrorException("Input indices must be contained in the ITensor"))

  Ris = difference(inds(A),Lis)
  #TODO: check if A is already ordered properly
  #and avoid doing this permute, since it makes a copy
  #AND/OR use svd!() to overwrite the data of A to save memory
  A = permute(A,Lis...,Ris...)
  Uis,Ustore,Sis,Sstore,Vis,Vstore = storage_svd(store(A),Lis,Ris;kwargs...)
  return ITensor(Uis,Ustore),ITensor(Sis,Sstore),ITensor(Vis,Vstore)
end

