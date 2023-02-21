using ITensors
using ITensors.ITensorNetworkMaps

struct InfTN
  data::Vector{ITensor}
end
Base.length(m::InfTN) = length(m.data)
Base.copy(m::InfTN) = InfTN(copy(m.data))
Base.reverse(m::InfTN) = InfTN(reverse(m.data))
Base.iterate(m::InfTN, args...) = iterate(m.data, args...)
Base.getindex(m::InfTN, args...) = getindex(m.data, args...)
Base.setindex!(m::InfTN, args...) = setindex!(m.data, args...)

function infmps(N; χ⃗, d)
  n⃗ = 1:N
  e⃗ = [n => mod1(n + 1, N) for n in 1:N]
  linkindex(χ⃗, e) = Index(χ⃗[e], "l=$(e[1])↔$(e[2])")
  l⃗ = Dict([e .=> linkindex(χ⃗, e) for e in e⃗])
  s⃗ = [Index(d, "s=$n") for n in n⃗]
  neigbhors(n, N) = [mod1(n - 1, N) => n, n => mod1(n + 1, N)]
  return InfTN([ITensor(getindex.(Ref(l⃗), neigbhors(n, N))..., s⃗[n]) for n in n⃗])
end

ITensors.dag(tn::InfTN) = InfTN(dag.(tn.data))
function ITensors.prime(::typeof(linkinds), tn::InfTN)
  tn_p = copy(tn)
  N = length(tn)
  for i in 1:N, j in (i + 1):N
    l = commoninds(tn[i], tn[j])
    tn_p[i] = prime(tn_p[i]; inds=l)
    tn_p[j] = prime(tn_p[j]; inds=l)
  end
  return tn_p
end

interleave(x::Vector, y::Vector) = permutedims([x y])[:]

function ITensors.linkind(tn::InfTN, e)
  return commonind(tn[e[1]], tn[e[2]])
end

function transfer_matrix(ψ::InfTN)
  N = length(ψ)
  ψ′ = prime(linkinds, dag(ψ))
  tn = interleave(reverse(ψ.data), reverse(ψ′.data))
  right_inds = [linkind(ψ, N => 1), linkind(ψ′, N => 1)]
  left_inds = [linkind(ψ, N => 1), linkind(ψ′, N => 1)]
  T = ITensorNetworkMap(tn; input_inds=right_inds, output_inds=left_inds)
  return T
end

function transfer_matrices(ψ::InfTN)
  N = length(ψ)
  ψ′ = prime(linkinds, dag(ψ))
  # Build from individual transfer matrices
  T⃗ = Vector{ITensorNetworkMap}(undef, N)
  for n in 1:N
    n⁺¹ = mod1(n + 1, N)
    n⁻¹ = mod1(n - 1, N)
    right_inds = [linkind(ψ, n => n⁺¹), linkind(ψ′, n => n⁺¹)]
    left_inds = [linkind(ψ, n⁻¹ => n), linkind(ψ′, n⁻¹ => n)]
    T⃗[n] = ITensorNetworkMap([ψ[n], ψ′[n]]; input_inds=right_inds, output_inds=left_inds)
  end
  return T⃗
end
