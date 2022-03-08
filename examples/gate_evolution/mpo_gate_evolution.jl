using ITensors

import ITensors: op

function op(::OpName"expτSS", ::SiteType"S=1/2", s1::Index, s2::Index; τ)
  h =
    1 / 2 * op("S+", s1) * op("S-", s2) +
    1 / 2 * op("S-", s1) * op("S+", s2) +
    op("Sz", s1) * op("Sz", s2)
  return exp(τ * h)
end

function main(; N=10, cutoff=1E-8, δt=0.1, ttotal=5.0)
  # Compute the number of steps to do
  Nsteps = Int(ttotal / δt)

  # Make an array of 'site' indices
  s = siteinds("S=1/2", N; conserve_qns=true)

  # Make gates (1,2),(2,3),(3,4),...
  gates = ops([("expτSS", (n, n + 1), (τ=-δt * im / 2,)) for n in 1:(N - 1)], s)

  # Include gates in reverse order too
  # (N,N-1),(N-1,N-2),...
  append!(gates, reverse(gates))

  # Function that measures <Sz> on site n
  function measure_Sz(psi::MPS, n)
    psi = orthogonalize(psi, n)
    sn = siteind(psi, n)
    Sz = scalar(dag(prime(psi[n], "Site")) * op("Sz", sn) * psi[n])
    return real(Sz)
  end

  # Initialize psi to be a product state (alternating up and down)
  psi0 = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

  c = div(N, 2)

  # Compute and print initial <Sz> value
  t = 0.0
  Sz = measure_Sz(psi0, c)
  println("$t $Sz")

  # Do the time evolution by applying the gates
  # for Nsteps steps
  psi = psi0
  for step in 1:Nsteps
    psi = apply(gates, psi; cutoff)
    t += δt
    Sz = measure_Sz(psi, c)
    println("$t $Sz")
  end

  # Now do the same evolution with an MPO
  rho0 = MPO(psi0)
  rho = rho0
  for step in 1:Nsteps
    rho = apply(gates, rho; cutoff, apply_dag=true)
    t += δt
  end
  @show inner(psi, rho, psi)
  @show inner(psi, psi)
  @show tr(rho)

  return nothing
end
