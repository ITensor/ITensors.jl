export orthog!,
       recursiveSVD

function orthog!(M::AbstractMatrix{T};
                 npass::Int=2, rng::MersenneTwister= Random.GLOBAL_RNG) where {T}
  nkeep = min(size(M)...)
  dots = zeros(T,nkeep)
  for i=1:nkeep
    coli = view(M,:,i)
    nrm = norm(coli)
    if nrm < 1E-10
      rand!(rng, coli)
      nrm = norm(coli)
    end
    coli ./= nrm
    (i==1) && continue

    Mcols = view(M,:,1:i-1)
    dotsref = view(dots,1:i-1)
    for pass=1:npass
      mul!(dotsref,Mcols',coli)
      #BLAS.gemv!('N',1.0,Mcols,dotsref,-1.0,coli)
      coli .-= Mcols*dotsref
      nrm = norm(coli)
      if nrm < 1E-3 #orthog is suspect
        pass = pass-1
      end
      if nrm < 1E-10
        rand!(rng, coli)
        nrm = norm(coli)
      end
      coli ./= nrm
    end
  end
end

function pos_sqrt(x::Float64)::Float64
  (x < 0.0) && return 0.0
  return sqrt(x)
end

function checkSVDDone(S::Vector{T},
                      thresh::Float64) where {T}
  N = length(S)
  (N <= 1 || thresh < 0.0) && return (true,1)
  S1t = S[1]*thresh
  start = 2
  while start <= N
    (S[start] < S1t) && break
    start += 1
  end
  if start >= N
    return (true,N)
  end
  return (false,start)
end

function recursiveSVD(M::AbstractMatrix{T};
                      thresh::Float64=1E-3,
                      north_pass::Int=2, 
                      rng::MersenneTwister=Random.GLOBAL_RNG) where {T}
  Mr,Mc = size(M)

  if Mr > Mc
    V,S,U = recursiveSVD(transpose(M), rng=rng)
    conj!(U)
    conj!(V)
    return U,S,V
  end

  #rho = BLAS.gemm('N','T',-1.0,M,M) #negative to sort eigenvalues greatest to smallest
  rho = -M*M' #negative to sort eigenvalues in decreasing order
  D,U = eigen(Hermitian(rho),1:size(rho,1))

  Nd = length(D)
  for n=1:Nd
    D[n] = pos_sqrt(-D[n])
  end

  V = M'*U
  orthog!(V,npass=north_pass, rng=rng)

  (done,start) = checkSVDDone(D,thresh)

  done && return U,D,V

  u = view(U,:,start:Nd)
  v = view(V,:,start:Nd)

  b = u'*(M*v)
  bu,bd,bv = recursiveSVD(b,
                          thresh=thresh,
                          north_pass=north_pass, rng = rng)

  u .= u*bu
  v .= v*bv
  view(D,start:Nd) .= bd
  
  return U,D,V
end
