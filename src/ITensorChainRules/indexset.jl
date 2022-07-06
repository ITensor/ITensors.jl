function setinds_pullback(ȳ, x, a...)
  x̄ = ITensors.setinds(ȳ, inds(x))
  ā = map_notangent(a)
  return (NoTangent(), x̄, ā...)
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

function inv_op(::typeof(addtags), x, args...; kwargs...)
  return removetags(x, args...; kwargs...)
end

function inv_op(::typeof(removetags), x, args...; kwargs...)
  return addtags(x, args...; kwargs...)
end

function inv_op(::typeof(replacetags), x, n1n2::Pair; kwargs...)
  return replacetags(x, reverse(n1n2); kwargs...)
end

_check_inds(x::ITensor, y::ITensor) = hassameinds(x, y)
_check_inds(x::MPS, y::MPS) = hassameinds(siteinds, x, y)
_check_inds(x::MPO, y::MPO) = hassameinds(siteinds, x, y)

for fname in (
  :prime, :setprime, :noprime, :replaceprime, :addtags, :removetags, :replacetags, :settags
)
  @eval begin
    function ChainRulesCore.rrule(f::typeof($fname), x::Union{MPS,MPO}, a...; kwargs...)
      y = f(x, a...; kwargs...)
      function f_pullback(ȳ)
        x̄ = inv_op(f, unthunk(ȳ), a...; kwargs...)
        if !_check_inds(x, x̄)
          error(
            "Trying to differentiate function `$f` with arguments $a and keyword arguments $kwargs. The forward pass indices $(inds(x)) do not match the reverse pass indices $(inds(x̄)). Likely this is because the priming/tagging operation you tried to perform is not invertible. Please write your code in a way where the index manipulation operation you are performing is invertible. For example, `prime(A::ITensor)` is invertible, with an inverse `prime(A, -1)`. However, `noprime(A)` is in general not invertible since the information about the prime levels of the original tensor are lost. Instead, you might try `prime(A, -1)` or `replaceprime(A, 1 => 0)` which are invertible.",
          )
        end
        ā = map_notangent(a)
        return (NoTangent(), x̄, ā...)
      end
      return y, f_pullback
    end
  end
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
      function f_pullback(ȳ)
        uȳ = unthunk(ȳ)
        x̄ = replaceinds(uȳ, inds(y), inds(x))
        ā = map_notangent(a)
        return (NoTangent(), x̄, ā...)
      end
      return y, f_pullback
    end
  end
end

function ChainRulesCore.rrule(::typeof(adjoint), x::ITensor)
  y = x'
  function adjoint_pullback(ȳ)
    uȳ = unthunk(ȳ)
    x̄ = replaceinds(uȳ, inds(y), inds(x))
    return (NoTangent(), x̄)
  end
  return y, adjoint_pullback
end

function ChainRulesCore.rrule(::typeof(adjoint), x::Union{MPS,MPO})
  y = x'
  function adjoint_pullback(ȳ)
    x̄ = inv_op(prime, ȳ)
    return (NoTangent(), x̄)
  end
  return y, adjoint_pullback
end

@non_differentiable permute(::Indices, ::Indices)
