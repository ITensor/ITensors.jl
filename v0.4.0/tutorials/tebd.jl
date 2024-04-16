using ITensors

let
  N = 100
  cutoff = 1E-8
  tau = 0.1
  ttotal = 5.0

  # Make an array of 'site' indices
  s = siteinds("S=1/2", N; conserve_qns=true)

  # Make gates (1,2),(2,3),(3,4),...
  gates = ITensor[]
  for j in 1:(N - 1)
    s1 = s[j]
    s2 = s[j + 1]
    hj =
      op("Sz", s1) * op("Sz", s2) +
      1 / 2 * op("S+", s1) * op("S-", s2) +
      1 / 2 * op("S-", s1) * op("S+", s2)
    Gj = exp(-1.0im * tau / 2 * hj)
    push!(gates, Gj)
  end
  # Include gates in reverse order too
  # (N,N-1),(N-1,N-2),...
  append!(gates, reverse(gates))

  # Initialize psi to be a product state (alternating up and down)
  psi = MPS(s, n -> isodd(n) ? "Up" : "Dn")

  c = div(N, 2) # center site

  # Compute and print <Sz> at each time step
  # then apply the gates to go to the next time
  for t in 0.0:tau:ttotal
    Sz = expect(psi, "Sz"; sites=c)
    println("$t $Sz")

    t ≈ ttotal && break

    psi = apply(gates, psi; cutoff)
    normalize!(psi)
  end

  return nothing
end
