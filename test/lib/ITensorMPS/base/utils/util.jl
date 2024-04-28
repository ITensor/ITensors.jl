using ITensors
using Random

using ITensors: AbstractMPS

function fill_trivial_coefficients(ψ)
  return ψ isa AbstractMPS ? (1, ψ) : ψ
end

function inner_add(α⃗ψ⃗::Tuple{<:Number,<:MPST}...) where {MPST<:AbstractMPS}
  Nₘₚₛ = length(α⃗ψ⃗)
  α⃗ = first.(α⃗ψ⃗)
  ψ⃗ = last.(α⃗ψ⃗)
  N⃡ = (conj(α⃗[i]) * α⃗[j] * inner(ψ⃗[i], ψ⃗[j]) for i in 1:Nₘₚₛ, j in 1:Nₘₚₛ)
  return sum(N⃡)
end

inner_add(ψ⃗...) = inner_add(fill_trivial_coefficients.(ψ⃗)...)

# TODO: this is no longer needed, use randomMPS
function makeRandomMPS(sites; chi::Int=4)::MPS
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  l = [Index(chi, "Link,l=$n") for n in 1:(N - 1)]
  for n in 1:N
    s = sites[n]
    if n == 1
      v[n] = randomITensor(l[n], s)
    elseif n == N
      v[n] = randomITensor(l[n - 1], s)
    else
      v[n] = randomITensor(l[n - 1], l[n], s)
    end
    normalize!(v[n])
  end
  return MPS(v, 0, N + 1)
end

function makeRandomMPO(sites; chi::Int=4)::MPO
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  l = [Index(chi, "Link,l=$n") for n in 1:(N - 1)]
  for n in 1:N
    s = sites[n]
    if n == 1
      v[n] = ITensor(l[n], s, s')
    elseif n == N
      v[n] = ITensor(l[n - 1], s, s')
    else
      v[n] = ITensor(l[n - 1], s, s', l[n])
    end
    randn!(v[n])
    normalize!(v[n])
  end
  return MPO(v, 0, N + 1)
end
