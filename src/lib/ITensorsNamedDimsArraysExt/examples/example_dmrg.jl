using Adapt: adapt
using ITensorMPS: MPO, OpSum, dmrg, random_mps, siteinds
using ITensors.ITensorsNamedDimsArraysExt: to_nameddimsarray

function main(; n, conserve_qns=false, nsweeps=3, cutoff=1e-4, arraytype=Array)
  s = siteinds("S=1/2", n; conserve_qns)
  ℋ = OpSum()
  ℋ = sum(j -> ("S+", j, "S-", j + 1), 1:(n - 1); init=ℋ)
  ℋ = sum(j -> ("S-", j, "S+", j + 1), 1:(n - 1); init=ℋ)
  ℋ = sum(j -> ("Sz", j, "Sz", j + 1), 1:(n - 1); init=ℋ)
  H = MPO(ℋ, s)
  ψ₀ = random_mps(s, j -> isodd(j) ? "↑" : "↓")

  H = adapt(arraytype, H)
  ψ = adapt(arraytype, ψ₀)
  e, ψ = dmrg(H, ψ; nsweeps, cutoff)

  Hₙₐ = to_nameddimsarray(H)
  Hₙₐ = adapt(arraytype, Hₙₐ)
  ψₙₐ = to_nameddimsarray(ψ₀)
  ψₙₐ = adapt(arraytype, ψₙₐ)
  eₙₐ, ψₙₐ = dmrg(Hₙₐ, ψₙₐ; nsweeps, cutoff)
  return (; e, eₙₐ)
end
