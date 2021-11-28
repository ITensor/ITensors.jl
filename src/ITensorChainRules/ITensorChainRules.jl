module ITensorChainRules

using ChainRulesCore
using ..ITensors

include("zygoterules.jl")

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

function inv_op(::typeof(prime), x::ITensor, n::Integer=1; kwargs...)
  return prime(x, -n; kwargs...)
end

function inv_op(::typeof(replaceprime), x::ITensor, n1n2::Pair; kwargs...)
  return replaceprime(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(swapprime), x::ITensor, n1n2::Pair; kwargs...)
  return swapprime(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(addtags), x::ITensor, args...; kwargs...)
  return removetags(x, args...; kwargs...)
end

function inv_op(::typeof(removetags), x::ITensor, args...; kwargs...)
  return addtags(x, args...; kwargs...)
end

function inv_op(::typeof(replacetags), x::ITensor, n1n2::Pair; kwargs...)
  return replacetags(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(swaptags), x::ITensor, n1n2::Pair; kwargs...)
  return swaptags(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(replaceind), x::ITensor, n1n2::Pair; kwargs...)
  return replaceind(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(replaceinds), x::ITensor, n1n2::Pair; kwargs...)
  return replaceinds(x, reverse(n1n2); kwargs...)
end

function inv_op(::typeof(swapind), x::ITensor, args...; kwargs...)
  return swapind(x, reverse(args)...; kwargs...)
end

function inv_op(::typeof(swapinds), x::ITensor, args...; kwargs...)
  return swapinds(x, reverse(args)...; kwargs...)
end

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
    function ChainRulesCore.rrule(f::typeof($fname), x::ITensor, a...; kwargs...)
      y = f(x, a...; kwargs...)
      function f_pullback(ȳ)
        x̄ = inv_op(f, unthunk(ȳ), a...; kwargs...)
        if !hassameinds(x, x̄)
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
    # TODO: define `dag(::Number)` to help with generic code
    x̄2 = dag(ITensor(x1)) * ȳ
    return (NoTangent(), x̄1[], x̄2)
  end
  return y, contract_pullback
end

function ChainRulesCore.rrule(::typeof(*), x1::ITensor, x2::Number)
  y = x1 * x2
  function contract_pullback(ȳ)
    x̄1 = ȳ * dag(ITensor(x2))
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

function ChainRulesCore.rrule(::typeof(ITensor), x::Array{<:Number}, a::Index...)
  y = ITensor(x, a...)
  function ITensor_pullback(ȳ)
    # TODO: define `Array(::ITensor)` directly
    x̄ = Array(unthunk(ȳ), inds(unthunk(ȳ))...)
    ā = broadcast(_ -> NoTangent(), a)
    return (NoTangent(), x̄, ā...)
  end
  return y, ITensor_pullback
end

function ChainRulesCore.rrule(::typeof(ITensor), x::Number)
  y = ITensor(x)
  function ITensor_pullback(ȳ)
    x̄ = ȳ[]
    return (NoTangent(), x̄)
  end
  return y, ITensor_pullback
end

function ChainRulesCore.rrule(::typeof(dag), x::ITensor)
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

end
