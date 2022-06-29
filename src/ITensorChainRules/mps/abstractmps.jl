function rrule(::Type{T}, x::Vector{<:ITensor}; kwargs...) where {T<:Union{MPS,MPO}}
  y = T(x; kwargs...)
  function T_pullback(ȳ)
    ȳtensors = ȳ.data
    n = length(ȳtensors)
    envL = [ȳtensors[1] * dag(x[1])]
    envR = [ȳtensors[n] * dag(x[n])]
    for j in 2:(n - 1)
      push!(envL, envL[j - 1] * ȳtensors[j] * dag(x[j]))
      push!(envR, envR[j - 1] * ȳtensors[n + 1 - j] * dag(x[n + 1 - j]))
    end

    x̄ = ITensor[]
    push!(x̄, ȳtensors[1] * envR[n - 1])
    for j in 2:(n - 1)
      push!(x̄, envL[j - 1] * ȳtensors[j] * envR[n - j])
    end
    push!(x̄, envL[n - 1] * ȳtensors[n])
    return (NoTangent(), x̄)
  end
  return y, T_pullback
end

function rrule(::typeof(inner), x1::T, x2::T; kwargs...) where {T<:Union{MPS,MPO}}
  if !hassameinds(siteinds, x1, x2)
    error(
      "Taking gradients of `inner(::MPS, ::MPS)` is not supported if the site indices of the input MPS don't match. If you input `inner(x, Ay)` where `Ay` is the result of something like `contract(A::MPO, y::MPS)`, try `inner(x', Ay)` or `inner(x, replaceprime(Ay, 1 => 0))`instead.",
    )
  end
  y = inner(x1, x2)
  function inner_pullback(ȳ)
    x̄1 = dag(ȳ) * x2
    # `dag` of `x1` gets reversed by `inner`
    x̄2 = x1 * ȳ
    return (NoTangent(), x̄1, x̄2)
  end
  return y, inner_pullback
end

# TODO: Define a more general version in ITensors.jl
function _contract(::Type{ITensor}, ψ::Union{MPS,MPO}, ϕ::Union{MPS,MPO}; kwargs...)
  T = ITensor(1)
  for n in 1:length(ψ)
    T = T * ψ[n] * ϕ[n]
  end
  return T
end

function _contract(::Type{MPO}, ψ::MPS, ϕ::MPS; kwargs...)
  ψmat = convert(MPO, ψ)
  ϕmat = convert(MPO, ϕ)
  return contract(ψmat, ϕmat; kwargs...)
end

function rrule(::typeof(apply), x1::Vector{ITensor}, x2::Union{MPS,MPO}; kwargs...)
  N = length(x1) + 1
  apply_dag = x2 isa MPO ? get(kwargs, :apply_dag, false) : nothing

  # Apply circuit and store intermediates in the forward direction
  x1x2 = Vector{typeof(x2)}(undef, N)
  x1x2[1] = x2
  for n in 2:N
    x1x2[n] = apply(x1[n - 1], x1x2[n - 1]; move_sites_back=true, kwargs...)
  end
  y = x1x2[end]

  function apply_pullback(ȳ)
    x1x2dag = dag.(x1x2)
    x1dag = [swapprime(dag(x), 0 => 1) for x in x1]

    # Apply circuit and store intermediates in the reverse direction
    x1dag_ȳ = Vector{typeof(x2)}(undef, N)
    x1dag_ȳ[end] = ȳ
    for n in (N - 1):-1:1
      x1dag_ȳ[n] = apply(x1dag[n], x1dag_ȳ[n + 1]; move_sites_back=true, kwargs...)
    end

    x̄1 = similar(x1)
    for n in 1:length(x1)
      # check if it's not a noisy gate (rank-3 tensor)
      if iseven(length(inds(x1[n])))
        gateinds = inds(x1[n]; plev=0)
        if x2 isa MPS
          ξ̃ = prime(x1dag_ȳ[n + 1], gateinds)
          ϕ̃ = x1x2dag[n]
        else
          # apply U on one side of the MPO
          if apply_dag
            ϕ̃ = swapprime(x1x2dag[n], 0 => 1)
            ϕ̃ = apply(x1[n], ϕ̃; move_sites_back=true, apply_dag=false)
            ϕ̃ = mapprime(ϕ̃, 1 => 2, 0 => 1)
            ϕ̃ = replaceprime(ϕ̃, 1 => 0; inds=gateinds')
            ξ̃ = 2 * dag(x1dag_ȳ[n + 1])'
          else
            ϕ̃ = mapprime(x1x2dag[n], 0 => 2)
            ϕ̃ = replaceprime(ϕ̃, 1 => 0; inds=gateinds')
            ξ̃ = mapprime(x1dag_ȳ[n + 1], 0 => 2)
          end
        end
        x̄1[n] = _contract(ITensor, ξ̃, ϕ̃)
      else
        s = inds(x1[n])
        x̄1[n] = itensor(zeros(dim.(s)), s...)
      end
    end
    x̄2 = x1dag_ȳ[end]
    return (NoTangent(), x̄1, x̄2)
  end
  return y, apply_pullback
end
