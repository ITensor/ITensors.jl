
function checkSVDDone(S::AbstractArray, thresh::Float64)
  N = length(S)
  (N <= 1 || thresh < 0.0) && return (true, 1)
  S1t = S[1] * thresh
  start = 2
  while start <= N
    (S[start] < S1t) && break
    start += 1
  end
  if start >= N
    return (true, N)
  end
  return (false, start)
end

function svd_recursive(M::AbstractMatrix; thresh::Float64=1E-3, north_pass::Int=2)
  Mr, Mc = size(M)
  if Mr > Mc
    V, S, U = svd_recursive(transpose(M))
    conj!(U)
    conj!(V)
    return U, S, V
  end

  #rho = BLAS.gemm('N','T',-1.0,M,M) #negative to sort eigenvalues greatest to smallest
  rho = -M * M' #negative to sort eigenvalues in decreasing order
  D, U = eigen(Hermitian(rho))

  Nd = length(D)

  V = M' * U

  V, R = qr_positive(V)
  for n in 1:Nd
    D[n] = R[n, n]
  end

  (done, start) = checkSVDDone(D, thresh)

  done && return U, D, V

  u = view(U, :, start:Nd)
  v = view(V, :, start:Nd)

  b = u' * (M * v)
  bu, bd, bv = svd_recursive(b; thresh=thresh, north_pass=north_pass)

  u .= u * bu
  v .= v * bv
  view(D, start:Nd) .= bd

  return U, D, V
end

# TODO: maybe move to another location?
# Include options for other svd algorithms
function polar(M::AbstractMatrix)
  U, S, V = svd(M) # calls LinearAlgebra.svd(_)
  return U * V', V * Diagonal(S) * V'
end
