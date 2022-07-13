using ITensors
using Printf

#=

This example code implements the minimally entangled typical thermal state (METTS).
For more information on METTS, see the following references:
- "Minimally entangled typical quantum states at finite temperature", Steven R. White,
  Phys. Rev. Lett. 102, 190601 (2009)
  and arxiv:0902.4475 (https://arxiv.org/abs/0902.4475)
- "Minimally entangled typical thermal state algorithms", E M Stoudenmire and Steven R White,
  New Journal of Physics 12, 055026 (2010) https://doi.org/10.1088/1367-2630/12/5/055026

=#

function ITensors.op(::OpName"expτSS", ::SiteType"S=1/2", s1::Index, s2::Index; τ)
  h =
    1 / 2 * op("S+", s1) * op("S-", s2) +
    1 / 2 * op("S-", s1) * op("S+", s2) +
    op("Sz", s1) * op("Sz", s2)
  return exp(τ * h)
end

"""
Given a Vector of numbers, returns
the average and the standard error
(= the width of distribution of the numbers)
"""
function avg_err(v::Vector)
  N = length(v)
  avg = v[1] / N
  avg2 = v[1]^2 / N
  for j in 2:N
    avg += v[j] / N
    avg2 += v[j]^2 / N
  end
  return avg, √((avg2 - avg^2) / N)
end

function main(; N=10, cutoff=1E-8, δτ=0.1, beta=2.0, NMETTS=3000, Nwarm=10)

  # Make an array of 'site' indices
  s = siteinds("S=1/2", N)

  # Make gates (1,2),(2,3),(3,4),...
  gates = ops([("expτSS", (n, n + 1), (τ=-δτ / 2,)) for n in 1:(N - 1)], s)
  # Include gates in reverse order to complete Trotter formula
  append!(gates, reverse(gates))

  # Make y-rotation gates to use in METTS collapses
  Ry_gates = ops([("Ry", n, (θ=π / 2,)) for n in 1:N], s)

  # Arbitrary initial state
  psi = randomMPS(s)

  # Make H for measuring the energy
  terms = OpSum()
  for j in 1:(N - 1)
    terms += 1 / 2, "S+", j, "S-", j + 1
    terms += 1 / 2, "S-", j, "S+", j + 1
    terms += "Sz", j, "Sz", j + 1
  end
  H = MPO(terms, s)

  # Make τ_range and check δτ is commensurate
  τ_range = δτ:δτ:(beta / 2)
  if norm(length(τ_range) * δτ - beta / 2) > 1E-10
    error("Time step δτ=$δτ not commensurate with beta/2=$(beta/2)")
  end

  energies = Float64[]

  for step in 1:(Nwarm + NMETTS)
    if step <= Nwarm
      println("Making warmup METTS number $step")
    else
      println("Making METTS number $(step-Nwarm)")
    end

    # Do the time evolution by applying the gates
    for τ in τ_range
      psi = apply(gates, psi; cutoff)
      normalize!(psi)
    end

    # Measure properties after >= Nwarm 
    # METTS have been made
    if step > Nwarm
      energy = inner(psi', H, psi)
      push!(energies, energy)
      @printf("  Energy of METTS %d = %.4f\n", step - Nwarm, energy)
      a_E, err_E = avg_err(energies)
      @printf(
        "  Estimated Energy = %.4f +- %.4f  [%.4f,%.4f]\n",
        a_E,
        err_E,
        a_E - err_E,
        a_E + err_E
      )
    end

    # Measure in X or Z basis on alternating steps
    if step % 2 == 1
      psi = apply(Ry_gates, psi)
      samp = sample!(psi)
      new_state = [samp[j] == 1 ? "X+" : "X-" for j in 1:N]
    else
      samp = sample!(psi)
      new_state = [samp[j] == 1 ? "Z+" : "Z-" for j in 1:N]
    end
    psi = productMPS(s, new_state)
  end

  return nothing
end
