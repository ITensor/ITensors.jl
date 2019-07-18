export orthog!,
       recursiveSVD

function orthog!(M::AbstractMatrix{T};
                 npass::Int=2) where {T}
  nkeep = min(size(M)...)
  dots = zeros(T,nkeep)
  for i=1:nkeep
    coli = view(M,:,i)
    nrm = norm(coli)
    if nrm < 1E-10
      rand!(coli)
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
        rand!(coli)
        nrm = norm(coli)
      end
      coli ./= nrm
    end
  end
end

function pos_sqrt(x::Float64)::Float64
  if x < 0.0
    return 0.0
  end
  return sqrt(x)
end

function recursiveSVD(M::AbstractMatrix{T}) where {T}
  Mr,Mc = size(M)

  if Mr > Mc
    return recursiveSVD(transpose(M))
  end

  #rho = BLAS.gemm('N','T',-1.0,M,M) #negative to sort eigenvalues greatest to smallest
  rho = -M*M' #negative to sort eigenvalues in decreasing order
  D,U = eigen(Hermitian(rho),1:size(rho,1))

  for n=1:length(D)
    D[n] = pos_sqrt(-D[n])
  end

  V = M'*U
  orthog!(V,npass=2)

  return U,D,V
end
