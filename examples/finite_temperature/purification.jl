using ITensors
import ITensors: op
using Printf

function op(::OpName"expτSS", ::SiteType"S=1/2", s1::Index, s2::Index; τ)
  h =
    1 / 2 * op("S+", s1) * op("S-", s2) +
    1 / 2 * op("S-", s1) * op("S+", s2) +
    op("Sz", s1) * op("Sz", s2)
  return exp(τ * h)
end

function main(; N=10, cutoff=1E-8, δτ=0.1, beta_max=2.0)
  L = 2 * N

  # Make an array of 'site' indices
  s = siteinds("S=1/2", L; conserve_qns=true)

  # Make gates (1,3),(3,5),(5,7),...
  # Only on physical (odd numbered) sites
  # Even sites are "ancilla" sites and have no Hamiltonian terms
  gates = ops([("expτSS", (n, n + 2), (τ=-δτ / 2,)) for n in 1:2:(L - 2)], s)
  # Include gates in reverse order to complete Trotter formula
  append!(gates, reverse(gates))

  # Initial state is infinite-temperature mixed state
  rho = MPO(s, "Id") ./ √2

  # Make H for measuring the energy
  terms = OpSum()
  for j in 1:2:(L - 2)
    terms += 1 / 2, "S+", j, "S-", j + 2
    terms += 1 / 2, "S-", j, "S+", j + 2
    terms += "Sz", j, "Sz", j + 2
  end
  H = MPO(terms, s)

  # Do the time evolution by applying the gates
  # for Nsteps steps
  for β in 0:δτ:beta_max
    energy = inner(rho, H)
    @printf("β = %.2f energy = %.8f\n", β, energy)
    rho = apply(gates, rho; cutoff)
    rho = rho / tr(rho)
  end

  return nothing
end
