#function rrule(::typeof(MPO), x::Vector{<:ITensor}; kwargs...)
#  y = MPO(x; kwargs...)
#  #@show y
#  function MPO_pullback(ȳ)
#    #@show ȳ
#    return ȳ.data
#    #ȳtensors = ȳ.data
#    #n = length(ȳtensors)
#    #envL = [ȳtensors[1] * dag(x[1]), ]
#    #envR = [ȳtensors[n] * dag(x[n]), ]
#    #for j in 2:n-1
#    #  push!(envL, envL[j-1] * ȳtensors[j] * dag(x[j]))
#    #  push!(envR, envR[j-1] * ȳtensors[n+1-j] * dag(x[n+1-j]))
#    #end
#    #x̄= ITensor[]
#    #push!(x̄, ȳtensors[1]  * envR[n-1])
#    #for j in 2:n-1
#    # push!(x̄, envL[j-1] * ȳtensors[j] * envR[n-j])
#    #end
#    #push!(x̄, envL[n-1] * ȳtensors[n])
#    #return (NoTangent(), x̄)
#  end
#  return y, MPO_pullback
#end

function rrule(::typeof(*), x1::MPO, x2::MPO; kwargs...)
  y = *(x1, x2; kwargs...)
  function contract_pullback(ȳ)
    x̄1 = *(ȳ, dag(x2); kwargs...)
    x̄2 = *(dag(x1), ȳ; kwargs...)
    return (NoTangent(), x̄1, x̄2)
  end
  return y, contract_pullback
end

function ChainRulesCore.rrule(::typeof(+), x1::MPO, x2::MPO; kwargs...)
  y = +(x1, x2; kwargs...)
  function add_pullback(ȳ)
    return (NoTangent(), ȳ, ȳ)
  end
  return y, add_pullback
end

function ChainRulesCore.rrule(::typeof(-), x1::MPO, x2::MPO; kwargs...)
  y = -(x1, x2; kwargs...)
  function add_pullback(ȳ)
    return (NoTangent(), ȳ, -ȳ)
  end
  return y, add_pullback
end

function rrule(::typeof(tr), x::MPO; kwargs...)
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

function rrule(::typeof(inner), x1::MPS, x2::MPO, x3::MPS; kwargs...)
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
