module ITensorChainRules

using ChainRulesCore
using ..ITensors

include("zygoterules.jl")

ITensors.dag(z::AbstractZero) = z

function ChainRulesCore.rrule(::typeof(getindex), x::ITensor, I...)
  y = getindex(x, I...)
  function getindex_pullback(ȳ)
    # TODO: add definition `ITensor(::Tuple{}) = ITensor()`
    # to ITensors.jl so no splatting is needed here.
    x̄ = ITensor(inds(x)...)
    x̄[I...] = unthunk(ȳ)
    Ī = broadcast_notangent(I)
    return (NoTangent(), x̄, Ī...)
  end
  return y, getindex_pullback
end

# Specialized version in order to avoid call to `setindex!`
# within the pullback, should be better for taking higher order
# derivatives in Zygote.
function ChainRulesCore.rrule(::typeof(getindex), x::ITensor)
  y = x[]
  function getindex_pullback(ȳ)
    x̄ = ITensor(unthunk(ȳ))
    return (NoTangent(), x̄)
  end
  return y, getindex_pullback
end

function setinds_pullback(ȳ, x, a...)
  x̄ = ITensors.setinds(ȳ, inds(x))
  ā = broadcast_notangent(a)
  return (NoTangent(), x̄, ā...)
end

function inv_op(f::Function, args...; kwargs...)
  return error(
    "Trying to differentiate `$f` but the inverse of the operation (`inv_op`) `$f` with arguments $args and keyword arguments $kwargs is not defined.",
  )
end

function inv_op(::typeof(prime), x, n::Integer=1; kwargs...)
  return prime(x, -n; kwargs...)
end

function inv_op(::typeof(replaceprime), x, n1n2::Pair; kwargs...)
  return replaceprime(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(swapprime), x, n1n2::Pair; kwargs...)
  return swapprime(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(addtags), x, args...; kwargs...)
  return removetags(x, args...; kwargs...)
end

function inv_op(::typeof(removetags), x, args...; kwargs...)
  return addtags(x, args...; kwargs...)
end

function inv_op(::typeof(replacetags), x, n1n2::Pair; kwargs...)
  return replacetags(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(swaptags), x, n1n2::Pair; kwargs...)
  return swaptags(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(replaceind), x, n1n2::Pair; kwargs...)
  return replaceind(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(replaceinds), x, n1n2::Pair; kwargs...)
  return replaceinds(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(swapind), x, args...; kwargs...)
  return swapind(x, reverse(args)...; kwargs...)
end

function inv_op(::typeof(swapinds), x, args...; kwargs...)
  return swapinds(x, reverse(args)...; kwargs...)
end

_check_inds(x::ITensor, y::ITensor) = hassameinds(x, y)
_check_inds(x::MPS, y::MPS) = hassameinds(siteinds, x, y)
_check_inds(x::MPO, y::MPO) = hassameinds(siteinds, x, y)

for fname in (
  :prime,
  :setprime,
  :noprime,
  :replaceprime,
  :swapprime,
  :addtags,
  :removetags,
  :replacetags,
  :settags,
  :swaptags,
  :replaceind,
  :replaceinds,
  :swapind,
  :swapinds,
)
  @eval begin
    function ChainRulesCore.rrule(
      f::typeof($fname), x::Union{ITensor,MPS,MPO}, a...; kwargs...
    )
      y = f(x, a...; kwargs...)
      function f_pullback(ȳ)
        x̄ = inv_op(f, unthunk(ȳ), a...; kwargs...)
        if !_check_inds(x, x̄)
          error(
            "Trying to differentiate function `$f` with arguments $a and keyword arguments $kwargs. The forward pass indices $(inds(x)) do not match the reverse pass indices $(inds(x̄)). Likely this is because the priming/tagging operation you tried to perform is not invertible. Please write your code in a way where the index manipulation operation you are performing is invertible. For example, `prime(A::ITensor)` is invertible, with an inverse `prime(A, -1)`. However, `noprime(A)` is in general not invertible since the information about the prime levels of the original tensor are lost. Instead, you might try `prime(A, -1)` or `replaceprime(A, 1 => 0)` which are invertible.",
          )
        end
        ā = broadcast_notangent(a)
        return (NoTangent(), x̄, ā...)
      end
      return y, f_pullback
    end
  end
end

# TODO: This is not being called by Zygote for some reason,
# using a Zygote overload directly instead. Figure out
# why, maybe raise an issue.
#function ChainRulesCore.rrule(::typeof(adjoint), x::ITensor)
#  y = prime(x)
#  function adjoint_pullback(ȳ)
#    return setinds_pullback(ȳ, x)
#  end
#  return y, adjoint_pullback
#end

# Special case for contracting a pair of ITensors
function ChainRulesCore.rrule(::typeof(*), x1::ITensor, x2::ITensor)
  y = x1 * x2
  function contract_pullback(ȳ)
    x̄1 = ȳ * dag(x2)
    x̄2 = dag(x1) * ȳ
    return (NoTangent(), x̄1, x̄2)
  end
  return y, contract_pullback
end

function ChainRulesCore.rrule(::typeof(*), x1::Number, x2::ITensor)
  y = x1 * x2
  function contract_pullback(ȳ)
    x̄1 = ȳ * dag(x2)
    x̄2 = dag(x1) * ȳ
    return (NoTangent(), x̄1[], x̄2)
  end
  return y, contract_pullback
end

function ChainRulesCore.rrule(::typeof(*), x1::ITensor, x2::Number)
  y = x1 * x2
  function contract_pullback(ȳ)
    x̄1 = ȳ * dag(x2)
    x̄2 = dag(x1) * ȳ
    return (NoTangent(), x̄1, x̄2[])
  end
  return y, contract_pullback
end

# TODO: use some contraction sequence optimization here
function ChainRulesCore.rrule(::typeof(*), x1::ITensor, x2::ITensor, xs::ITensor...)
  y = *(x1, x2, xs...)
  function contract_pullback(ȳ)
    tn = [x1, x2, xs...]
    N = length(tn)
    env_contracted = Vector{ITensor}(undef, N)
    for n in 1:length(tn)
      tn_left = tn[1:(n - 1)]
      # TODO: define contract([]) = ITensor(1.0)
      env_left = isempty(tn_left) ? ITensor(1.0) : contract(tn_left)
      tn_right = tn[reverse((n + 1):end)]
      env_right = isempty(tn_right) ? ITensor(1.0) : contract(tn_right)
      env_contracted[n] = dag(env_left) * ȳ * dag(env_right)
    end
    return (NoTangent(), env_contracted...)
  end
  return y, contract_pullback
end

function ChainRulesCore.rrule(::typeof(+), x1::ITensor, x2::ITensor)
  y = x1 + x2
  function add_pullback(ȳ)
    return (NoTangent(), ȳ, ȳ)
  end
  return y, add_pullback
end

function ChainRulesCore.rrule(::typeof(itensor), x::Array, a...)
  y = itensor(x, a...)
  function itensor_pullback(ȳ)
    uȳ = permute(unthunk(ȳ), a...)
    x̄ = reshape(array(uȳ), size(x))
    ā = broadcast_notangent(a)
    return (NoTangent(), x̄, ā...)
  end
  return y, itensor_pullback
end

function ChainRulesCore.rrule(::Type{ITensor}, x::Array{<:Number}, a...)
  y = ITensor(x, a...)
  function ITensor_pullback(ȳ)
    # TODO: define `Array(::ITensor)` directly
    uȳ = Array(unthunk(ȳ), a...)
    x̄ = reshape(uȳ, size(x))
    ā = broadcast_notangent(a)
    return (NoTangent(), x̄, ā...)
  end
  return y, ITensor_pullback
end

function ChainRulesCore.rrule(::Type{ITensor}, x::Number)
  y = ITensor(x)
  function ITensor_pullback(ȳ)
    x̄ = ȳ[]
    return (NoTangent(), x̄)
  end
  return y, ITensor_pullback
end

function ChainRulesCore.rrule(::typeof(dag), x)
  y = dag(x)
  function dag_pullback(ȳ)
    x̄ = dag(unthunk(ȳ))
    return (NoTangent(), x̄)
  end
  return y, dag_pullback
end

function ChainRulesCore.rrule(::typeof(permute), x::ITensor, a...)
  y = permute(x, a...)
  function permute_pullback(ȳ)
    x̄ = permute(unthunk(ȳ), inds(x))
    ā = broadcast_notangent(a)
    return (NoTangent(), x̄, ā...)
  end
  return y, permute_pullback
end

broadcast_notangent(a) = broadcast(_ -> NoTangent(), a)

@non_differentiable broadcast_notangent(::Any)
@non_differentiable Index(::Any...)
@non_differentiable delta(::Any...)
@non_differentiable dag(::Index)
@non_differentiable inds(::Any...)
@non_differentiable commoninds(::Any...)
@non_differentiable noncommoninds(::Any...)
@non_differentiable uniqueinds(::Any...)
@non_differentiable SiteType(::Any)
@non_differentiable ITensors._sitetypes(::Any)
@non_differentiable addtags(::TagSet, ::Any)
@non_differentiable ITensors.filter_inds_set_function(::Function, ::Function, ::Any...)
@non_differentiable ITensors.filter_inds_set_function(::Function, ::Any...)

#
# MPO/MPS
#

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

function ChainRulesCore.rrule(::typeof(apply), x1::Vector{ITensor}, x2::MPS; kwargs...)
  #y = apply(x1, x2; kwargs...)
  N = length(x1) + 1

  # Apply circuit and store intermediates in the forward direction
  x1x2 = Vector{MPS}(undef, N)
  x1x2[1] = x2
  for n in 2:N
    x1x2[n] = apply(x1[n - 1], x1x2[n - 1]; move_sites_back=true, kwargs...)
  end
  y = x1x2[end]

  function apply_pullback(ȳ)
    x1x2dag = dag.(x1x2)
    # Apply circuit and store intermediates in the reverse direction
    x1dag = [swapprime(dag(x), 0 => 1) for x in x1]

    x1dag_ȳ = Vector{MPS}(undef, N)
    x1dag_ȳ[end] = ȳ
    for n in (N - 1):-1:1
      x1dag_ȳ[n] = apply(x1dag[n], x1dag_ȳ[n + 1]; move_sites_back=true, kwargs...)
    end

    x̄1 = similar(x1)
    for n in 1:length(x1)
      x1dag_ȳ′ = prime(x1dag_ȳ[n + 1], inds(x1[n]; plev=0))
      x̄1[n] = _contract(ITensor, x1dag_ȳ′, x1x2dag[n])
    end
    x̄2 = x1dag_ȳ[end]

    return (NoTangent(), x̄1, x̄2)
  end
  return y, apply_pullback
end

function ChainRulesCore.rrule(::typeof(apply), x1::Vector{ITensor}, x2::MPO; kwargs...)
  apply_dag = kwargs[:apply_dag]
  N = length(x1) + 1
  # Apply circuit and store intermediates in the forward direction
  x1x2 = Vector{MPO}(undef, N)
  x1x2[1] = x2
  for n in 2:N
    x1x2[n] = apply(x1[n - 1], x1x2[n - 1]; move_sites_back=true, kwargs...)
  end
  y = x1x2[end]
  function apply_pullback(ȳ)
    x1x2dag = dag.(x1x2)
    x1dag_ȳ = Vector{MPO}(undef, N)
    x1dag_ȳ[end] = ȳ
    x1dag = [swapprime(dag(x), 0 => 1) for x in x1]
    for n in (N - 1):-1:1
      x1dag_ȳ[n] = apply(x1dag[n], x1dag_ȳ[n + 1]; move_sites_back=true, kwargs...)
    end

    x̄1 = similar(x1)
    for n in 1:length(x1)
      if iseven(length(inds(x1[n])))
        gateinds = inds(x1[n]; plev=1)
        if apply_dag
          ϕ̃ = swapprime(x1x2dag[n], 0 => 1)
          ϕ̃ = apply(x1[n], ϕ̃; move_sites_back=true, apply_dag=false)
          ϕ̃ = mapprime(ϕ̃, 1 => 2, 0 => 1)
          ϕ̃ = replaceprime(ϕ̃, 1 => 0; inds=gateinds)
          x̄1[n] = 2 * _contract(ITensor, dag(x1dag_ȳ[n + 1])', ϕ̃)
        else
          ϕ̃ = mapprime(x1x2dag[n], 0 => 2)
          ϕ̃ = replaceprime(ϕ̃, 1 => 0; inds=gateinds)
          ξ̃ = mapprime(x1dag_ȳ[n + 1], 0 => 2)
          x̄1[n] = _contract(ITensor, ξ̃, ϕ̃)
        end
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

function ChainRulesCore.rrule(::typeof(inner), x1::MPS, x2::MPO, x3::MPS; kwargs...)
  if !hassameinds(siteinds, x1, (x2, x3)) || !hassameinds(siteinds, x3, (x2, x1))
    error(
      "Taking gradients of `inner(x::MPS, A::MPO, y::MPS)` is not supported if the site indices of the input MPS and MPO don't match. Try using if you input `inner(x, A, y), try `inner(x', A, y)` instead.",
    )
  end

  y = inner(x1, x2, x3; kwargs...)
  function inner_pullback(ȳ)
    x̄1 = dag(ȳ) * contract(x2, x3; kwargs...)
    x̄2 = ȳ * dag(_contract(MPO, dag(x1), x3; kwargs...))
    x̄3 = contract(dag(x2), x1; kwargs...) * ȳ

    @assert siteinds(x1) == siteinds(x̄1)
    @assert hassameinds(siteinds, x2, x̄2)
    @assert siteinds(x3) == siteinds(x̄3)

    return (NoTangent(), x̄1, x̄2, x̄3)
  end
  return y, inner_pullback
end

function ChainRulesCore.rrule(::typeof(inner), x1::MPS, x2::MPS; kwargs...)
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

function ChainRulesCore.rrule(::typeof(*), x1::MPO, x2::MPO; kwargs...)
  y = *(x1, x2; kwargs...)
  function contract_pullback(ȳ)
    x̄1 = *(ȳ, dag(x2); kwargs...)
    x̄2 = *(dag(x1), ȳ; kwargs...)
    return (NoTangent(), x̄1, x̄2)
  end
  return y, contract_pullback
end

function ChainRulesCore.rrule(::typeof(tr), x::MPO; kwargs...)
  y = tr(x; kwargs...)
  function contract_pullback(ȳ)
    s = noprime(firstsiteinds(x))
    n = length(s)
    x̄ = ȳ * MPO(s, "Id")

    plev = get(kwargs, :plev, 0 => 1)
    for j in 1:n
      x̄[j] = mapprime(x̄[j], 0 => first(plev), 1 => last(plev))
    end
    return (NoTangent(), ȳ * x̄)
  end
  return y, contract_pullback
end

end
