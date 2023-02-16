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

# Based on https://discourse.julialang.org/t/lapackexception-1-while-svd-but-not-svdvals/23787
function make_illconditioned_matrix(T=5000)
  t = 0:(T - 1)
  f = LinRange(0, 0.5 - 1 / length(t) / 2, length(t) ÷ 2)
  y = sin.(t)
  function check_freq(f)
    zerofreq = findfirst(iszero, f)
    zerofreq !== nothing &&
      zerofreq != 1 &&
      throw(ArgumentError("If zero frequency is included it must be the first frequency"))
    return zerofreq
  end
  function get_fourier_regressor(t, f)
    zerofreq = check_freq(f)
    N = length(t)
    Nf = length(f)
    Nreg = zerofreq === nothing ? 2Nf : 2Nf - 1
    N >= Nreg || throw(ArgumentError("Too many frequency components $Nreg > $N"))
    A = zeros(N, Nreg)
    sinoffset = Nf
    for fn in 1:Nf
      if fn == zerofreq
        sinoffset = Nf - 1
      end
      for n in 1:N
        phi = 2π * f[fn] * t[n]
        A[n, fn] = cos(phi)
        if fn != zerofreq
          A[n, fn + sinoffset] = -sin(phi)
        end
      end
    end
    return A, zerofreq
  end
  A, z = get_fourier_regressor(t, f)
  return [A y]
end
