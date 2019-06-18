
function truncate!(P::Vector{Float64};
                   kwargs...
                  )::Tuple{Float64,Float64}
  maxdim::Int = get(kwargs,:maxdim,length(P))
  mindim::Int = get(kwargs,:mindim,1)
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
                   Linds...;
                   factorization=factorization)
  Lis = IndexSet(Linds...)
  !hasinds(A,Lis) && throw(ErrorException("Input indices must be contained in the ITensor"))
  Ris = uniqueinds(A,Lis)
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

qr(A::ITensor,Linds...) = factorize(A,Linds...;factorization=:QR)

polar(A::ITensor,Linds...) = factorize(A,Linds...;factorization=:polar)

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
* maxdim [Int]
* mindim [Int]
* cutoff [Float64]
* truncate [Bool]
"""
function svd(A::ITensor,
             Linds;
             kwargs...
            )
  Lis = IndexSet()
  for i in IndexSet(Linds)
    i ∈ inds(A) && push!(Lis,i)
  end
  Ris = uniqueinds(A,Lis)
  #TODO: check if A is already ordered properly
  #and avoid doing this permute, since it makes a copy
  #AND/OR use svd!() to overwrite the data of A to save memory

  global timer.svd_t += @elapsed begin
    A = permute(A,Lis...,Ris...)
    global timer.svd_store_t += @elapsed begin
    Uis,Ustore,Sis,Sstore,Vis,Vstore = storage_svd(store(A),Lis,Ris;kwargs...)
    end
    global timer.svd_store_c += 1
  end
  
  global timer.svd_c += 1
  U = ITensor(Uis,Ustore)
  S = ITensor(Sis,Sstore)
  V = ITensor(Vis,Vstore)
  u = commonindex(U,S)
  v = commonindex(S,V)
  return U,S,V,u,v
end

# TODO: add a version that automatically detects the IndexSets
# by matching based on tags
function eigen(A::ITensor,
               Linds,
               Rinds;
               truncate::Int=100,
               lefttags::String="Link,0",
               righttags::String="Link,1",
               matrixtype::Type{T}=Hermitian) where {T}
  Lis = IndexSet(Linds)
  Ris = IndexSet(Rinds)
  !hassameinds((Lis,Ris),A) && throw(ErrorException("Input indices must be contained in the ITensor"))

  #TODO: check if A is already ordered properly
  #and avoid doing this permute, since it makes a copy
  #AND/OR use svd!() to overwrite the data of A to save memory
  A = permute(A,(Lis,Ris))
  #TODO: More of the index analysis should be moved out of storage_eigen
  Uis,Ustore,Dis,Dstore = storage_eigen(store(A),Lis,Ris,matrixtype,truncate,lefttags,righttags)
  return ITensor(Uis,Ustore),ITensor(Dis,Dstore)
end

