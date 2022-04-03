using ITensors
import ITensors: op

function op(::OpName"expτSS", ::SiteType"S=1/2", s1::Index, s2::Index; τ)
  h =
    1 / 2 * op("S+", s1) * op("S-", s2) +
    1 / 2 * op("S-", s1) * op("S+", s2) +
    op("Sz", s1) * op("Sz", s2)
  return exp(τ * h)
end

function main(; N=10, cutoff=1E-8, δτ=0.05, beta_max=2.0)
  L = 2 * N

  # Make an array of 'site' indices
  s = siteinds("S=1/2", L; conserve_qns=true)

  # Make gates (1,3),(3,5),(5,7),...
  # Only on physical (odd numbered) sites
  # Even sites are "ancilla" sites and have no Hamiltonian terms
  gates = ops([("expτSS", (n, n + 2), (τ=-δτ / 2,)) for n in 1:2:(L - 2)], s)
  # Include gates in reverse order to complete Trotter formula
  append!(gates, reverse(gates))

  # Initialize psi to be a product of Bell pairs
  psi = MPS(L)
  for j in 1:2:(L - 1)
    s1 = s[j]
    s2 = s[j + 1]
    bell = ITensor([0 1/√2; 1/√2 0], s1, s2)
    # Restore MPS form
    U, S, V = svd(bell, s1)
    psi[j] = U * S
    psi[j + 1] = V
  end
  # Put in remaining link indices
  for j in 2:2:(L - 1)
    l = Index([QN() => 1], "n=$j")
    psi[j] *= dag(onehot(l => 1))
    psi[j + 1] *= onehot(l => 1)
  end

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
  for τ in 0:δτ:(beta_max / 2)
    En = inner(psi', H, psi)
    β = 2τ
    println("β = $β energy = $En")
    psi = apply(gates, psi; cutoff)
    normalize!(psi)
  end

  return nothing
end
