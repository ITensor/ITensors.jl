using ITensors
using ITensors.ITensorNetworkMaps

function transfer_matrix(ψ::MPS, n::Int)
  ψ′ = prime(linkinds, ψ)
  lⁿ = linkinds(ψ, n)
  lⁿ⁻¹ = linkinds(ψ, n - 1)
  input_inds_n = (lⁿ..., dag(lⁿ')...)
  output_inds_n = (lⁿ⁻¹..., dag(lⁿ⁻¹')...)
  return ITensorNetworkMap(
    [ψ[n], dag(ψ′[n])]; input_inds=input_inds_n, output_inds=output_inds_n
  )
end

function transfer_matrix(Hψ::Tuple{MPO,MPS}, n::Int)
  H, ψ = Hψ
  ψ′ = ψ'
  lⁿ = linkinds(ψ, n)
  lⁿ⁻¹ = linkinds(ψ, n - 1)
  hⁿ = linkinds(H, n)
  hⁿ⁻¹ = linkinds(H, n - 1)
  input_inds_n = (lⁿ..., hⁿ..., dag(lⁿ')...)
  output_inds_n = (lⁿ⁻¹..., hⁿ⁻¹..., dag(lⁿ⁻¹')...)
  return ITensorNetworkMap(
    [ψ[n], H[n], dag(ψ′[n])]; input_inds=input_inds_n, output_inds=output_inds_n
  )
end

function transfer_matrix(ψ, r::UnitRange)
  return prod([transfer_matrix(ψ, n) for n in r])
end

function cache(T::Vector{<:ITensorNetworkMap})
  v = Dict()
  v[0 => 1] = ITensor(1)
  for n in 1:N
    v[n => n + 1] = v[n - 1 => n] * T[n]
  end
  v[N + 1 => N] = ITensor(1)
  for n in reverse(1:N)
    v[n => n - 1] = T[n] * v[n + 1 => n]
  end
  return v
end

function hessian(H::MPO, cache, n::Int)
  return hessian(H, cache, n:n)
end

function hessian(H::MPO, cache, n⃗::UnitRange)
  n1 = first(n⃗)
  n2 = last(n⃗)
  Hᴸ = vᴴ[n1 - 1 => n1]
  Hᴿ = vᴴ[n2 + 1 => n2]
  Hⁿ⃗ = H[n⃗]
  return ITensorNetworkMap([Hᴸ, Hⁿ⃗..., Hᴿ])
end
