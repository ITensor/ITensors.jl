
function davidson(A,
                  phi0::ITensor;
                  kwargs...)::Tuple{Float64,ITensor}

  phi = copy(phi0)

  maxiter = get(kwargs,:maxiter,2)
  miniter = get(kwargs,:maxiter,1)
  errgoal = get(kwargs,:errgoal,1E-14)

  approx0 = 1E-12

  nrm = norm(phi)
  nrm < 1E-18 && (phi = randomITensor(inds(phi)))

  maxsize = size(A)[1]
  actual_maxiter = min(maxiter,maxsize-1)

  if dim(inds(phi)) != maxsize
    error("linear size of A and dimension of phi should match in davidson")
  end

  V = ITensor[phi]
  AV = ITensor[A(phi)]

  lambda = NaN
  last_lambda = NaN

  M = fill(0.0,(1,1))

  iter = 0
  for ni=1:actual_maxiter+1
    if ni == 1
      lambda = dot(V[1],AV[1])
      M[1,1] = lambda
      q = AV[1] - lambda*V[1];
    else #ni > 1
      F = eigen(Hermitian(M))
      lambda = F.values[1]
      #@show F.values
      #@show lambda
      u = F.vectors[:,1]
      phi = u[1]*V[1]
      q = u[1]*AV[1]
      for n=2:ni
        phi += u[n]*V[n]
        q   += u[n]*AV[n]
      end
      q -= lambda*phi
      #Fix sign
      if real(u[1]) < 0
        phi *= -1
        q *= -1
      end
    end

    qnorm = norm(q)

    converged = (qnorm < errgoal && abs(lambda-last_lambda) < errgoal) || (qnorm < max(approx0,errgoal*1E-3))

    last_lambda = lambda

    if (qnorm < 1E-20) || (converged && ni > miniter_) || (ni >= actual_maxiter)
      #@printf "done with davidson, ni=%d, qnorm=%.3E\n" ni qnorm
      break
    end

    Vq = fill(0.0,ni)
    Npass = 1
    pass = 1
    tot_pass = 0
    while pass <= Npass
      tot_pass += 1
      for k=1:ni
        Vq[k] = dot(V[k],q)
      end
      for k=1:ni
        q += -Vq[k]*V[k]
      end
      qnrm = norm(q)
      if qnrm < 1E-10 #orthog failure, try randomizing
        # TODO: put random recovery code here
        error("orthog failure")
      end
      q /= qnrm
      pass += 1
    end

    push!(V,q)
    push!(AV,A(q))

    newM = fill(0.0,(ni+1,ni+1))
    newM[1:ni,1:ni] = M
    for k=1:ni+1
      newM[k,ni+1] = dot(V[k],AV[ni+1])
      newM[ni+1,k] = conj(newM[k,ni+1])
    end
    M = newM
    #println("M = ")
    #display(M);println()

    #testM = fill(0.0,(ni+1,ni+1))
    #for i=1:ni+1,j=1:ni+1
    #  testM[i,j] = dot(V[i],AV[j])
    #end
    #println("testM = ")
    #display(testM);println()

    iter += 1

  end #for ni=1:actual_maxiter+1

  return lambda,phi

end

