
function get_vecs!((phi, q), M, V, AV, ni)
  F = eigen(Hermitian(M))
  lambda = F.values[1]
  u = F.vectors[:, 1]
  phi .= u[1] .* V[1]
  q .= u[1] .* AV[1]
  for n in 2:ni
    phi .+= u[n] .* V[n]
    q .+= u[n] .* AV[n]
  end
  q .-= lambda .* phi
  #Fix sign
  if real(u[1]) < 0.0
    phi .*= -1
    q .*= -1
  end
  return lambda
end

function orthogonalize!(q::ITensor, V, ni)
  q0 = copy(q)
  for k in 1:ni
    Vq0k = dot(V[k], q0)
    q .+= -Vq0k .* V[k]
  end
  qnrm = norm(q)
  if qnrm < 1E-10 #orthog failure, try randomizing
    randn!(q)
    qnrm = norm(q)
  end
  q .*= 1.0 / qnrm
  return nothing
end

function expand_krylov_space(M::Matrix{elT}, V, AV, ni) where {elT}
  newM = fill(zero(elT), (ni + 1, ni + 1))
  newM[1:ni, 1:ni] = M
  for k in 1:(ni + 1)
    newM[k, ni + 1] = dot(V[k], AV[ni + 1])
    newM[ni + 1, k] = conj(newM[k, ni + 1])
  end
  return newM
end

function davidson(A, phi0::ITensorT; kwargs...) where {ITensorT<:ITensor}
  elTA = eltype(A)
  elTphi = eltype(phi0)

  # if the matrix is complex and the starting vector is real,
  # that's not going to last long
  # and we're going to need phi to be complex
  # down when we reuse the storage in get_vecs!
  if !(elTA <: Real) && (elTphi <: Real)
    phi = complex(phi0)
  else
    phi = copy(phi0)
  end

  maxiter = get(kwargs, :maxiter, 2)
  miniter = get(kwargs, :miniter, 1)
  errgoal = get(kwargs, :errgoal, 1E-14)
  Northo_pass = get(kwargs, :Northo_pass, 1)

  approx0 = 1E-12

  nrm = norm(phi)
  if nrm < 1E-18
    phi_ = similar(phi)
    randn!(phi_)
    phi = phi_
    nrm = norm(phi)
  end
  phi .*= 1.0 / nrm

  maxsize = size(A)[1]
  actual_maxiter = min(maxiter, maxsize - 1)

  if dim(inds(phi)) != maxsize
    error("linear size of A and dimension of phi should match in davidson")
  end

  Aphi = A(phi)

  V = ITensorT[copy(phi)]
  AV = ITensorT[Aphi]

  last_lambda = NaN
  lambda::Float64 = real(dot(V[1], AV[1]))
  q = AV[1] - lambda * V[1]

  M = fill(elTA(lambda), (1, 1))

  for ni in 1:actual_maxiter
    qnorm = norm(q)

    errgoal_reached = (qnorm < errgoal && abs(lambda - last_lambda) < errgoal)
    small_qnorm = (qnorm < max(approx0, errgoal * 1E-3))
    converged = errgoal_reached || small_qnorm

    if (qnorm < 1E-20) || (converged && ni > miniter) #|| (ni >= actual_maxiter)
      #@printf "  done with davidson, ni=%d, qnorm=%.3E\n" ni qnorm
      break
    end

    last_lambda = lambda

    for pass in 1:Northo_pass
      orthogonalize!(q, V, ni)
    end

    Aq = A(q)

    push!(V, copy(q))
    push!(AV, Aq)

    M = expand_krylov_space(M, V, AV, ni)

    lambda = get_vecs!((phi, q), M, V, AV, ni + 1)
  end #for ni=1:actual_maxiter+1

  return lambda, phi
end
