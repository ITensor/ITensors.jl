function rrule(::typeof(getindex), x::ITensor, I...)
  y = getindex(x, I...)
  function getindex_pullback(ȳ)
    # TODO: add definition `ITensor(::Tuple{}) = ITensor()`
    # to ITensors.jl so no splatting is needed here.
    x̄ = ITensor(inds(x)...)
    x̄[I...] = unthunk(ȳ)
    Ī = map_notangent(I)
    return (NoTangent(), x̄, Ī...)
  end
  return y, getindex_pullback
end

# Specialized version in order to avoid call to `setindex!`
# within the pullback, should be better for taking higher order
# derivatives in Zygote.
function rrule(::typeof(getindex), x::ITensor)
  y = x[]
  function getindex_pullback(ȳ)
    x̄ = ITensor(unthunk(ȳ))
    return (NoTangent(), x̄)
  end
  return y, getindex_pullback
end

function rrule(::Type{ITensor}, x1::AllowAlias, x2::TensorStorage, x3)
  y = ITensor(x1, x2, x3)
  function ITensor_pullback(ȳ)
    x̄1 = NoTangent()
    x̄2 = ȳ.tensor.storage
    x̄3 = NoTangent()
    return (NoTangent(), x̄1, x̄2, x̄3)
  end
  return y, ITensor_pullback
end

function rrule(::Type{ITensor}, x1::AllowAlias, x2::Tensor)
  y = ITensor(x1, x2)
  function ITensor_pullback(ȳ)
    x̄1 = NoTangent()
    x̄2 = Tensor(x1, ȳ)
    return (NoTangent(), x̄1, x̄2)
  end
  return y, ITensor_pullback
end

function rrule(::Type{ITensor}, x1::Tensor)
  y = ITensor(x1)
  function ITensor_pullback(ȳ)
    x̄1 = Tensor(ȳ)
    return (NoTangent(), x̄1)
  end
  return y, ITensor_pullback
end

function rrule(::typeof(itensor), x1::Tensor)
  y = itensor(x1)
  function itensor_pullback(ȳ)
    x̄1 = tensor(ȳ)
    return (NoTangent(), x̄1)
  end
  return y, itensor_pullback
end

function rrule(f::Type{<:Tensor}, x1::AllowAlias, x2::ITensor)
  y = f(x1, x2)
  function Tensor_pullback(ȳ)
    x̄1 = NoTangent()
    x̄2 = ITensor(x1, ȳ)
    return (NoTangent(), x̄1, x̄2)
  end
  return y, Tensor_pullback
end

function rrule(f::Type{<:Tensor}, x1::ITensor)
  y = f(x1)
  function Tensor_pullback(ȳ)
    x̄1 = ITensor(ȳ)
    return (NoTangent(), x̄1)
  end
  return y, Tensor_pullback
end

function rrule(::typeof(tensor), x1::ITensor)
  y = tensor(x1)
  function tensor_pullback(ȳ)
    x̄1 = ITensor(typeof(storage(x1))(ȳ.storage.data), inds(x1))
    return (NoTangent(), x̄1)
  end
  return y, tensor_pullback
end

# Special case for contracting a pair of ITensors
function rrule(::typeof(contract), x1::ITensor, x2::ITensor)
  project_x1 = ProjectTo(x1)
  project_x2 = ProjectTo(x2)
  function contract_pullback(ȳ)
    x̄1 = project_x1(ȳ * dag(x2))
    x̄2 = project_x2(dag(x1) * ȳ)
    return (NoTangent(), x̄1, x̄2)
  end
  return x1 * x2, contract_pullback
end

@non_differentiable ITensors.optimal_contraction_sequence(::Any)

function rrule(::typeof(*), x1::Number, x2::ITensor)
  project_x1 = ProjectTo(x1)
  project_x2 = ProjectTo(x2)
  function contract_pullback(ȳ)
    x̄1 = project_x1((ȳ * dag(x2))[])
    x̄2 = project_x2(dag(x1) * ȳ)
    return (NoTangent(), x̄1, x̄2)
  end
  return x1 * x2, contract_pullback
end

function rrule(::typeof(*), x1::ITensor, x2::Number)
  project_x1 = ProjectTo(x1)
  project_x2 = ProjectTo(x2)
  function contract_pullback(ȳ)
    x̄1 = project_x1(ȳ * dag(x2))
    x̄2 = project_x2((dag(x1) * ȳ)[])
    return (NoTangent(), x̄1, x̄2)
  end
  return x1 * x2, contract_pullback
end

function rrule(::typeof(+), x1::ITensor, x2::ITensor)
  function add_pullback(ȳ)
    return (NoTangent(), ȳ, ȳ)
  end
  return x1 + x2, add_pullback
end

function rrule(::typeof(-), x1::ITensor, x2::ITensor)
  function subtract_pullback(ȳ)
    return (NoTangent(), ȳ, -ȳ)
  end
  return x1 - x2, subtract_pullback
end

function rrule(::typeof(-), x::ITensor)
  function minus_pullback(ȳ)
    return (NoTangent(), -ȳ)
  end
  return -x, minus_pullback
end

function rrule(::typeof(itensor), x::Array, a...)
  function itensor_pullback(ȳ)
    uȳ = permute(unthunk(ȳ), a...)
    x̄ = reshape(array(uȳ), size(x))
    ā = map_notangent(a)
    return (NoTangent(), x̄, ā...)
  end
  return itensor(x, a...), itensor_pullback
end

function rrule(::Type{ITensor}, x::Array{<:Number}, a...)
  function ITensor_pullback(ȳ)
    # TODO: define `Array(::ITensor)` directly
    uȳ = Array(unthunk(ȳ), a...)
    x̄ = reshape(uȳ, size(x))
    ā = map_notangent(a)
    return (NoTangent(), x̄, ā...)
  end
  return ITensor(x, a...), ITensor_pullback
end

function rrule(::Type{ITensor}, x::Number)
  function ITensor_pullback(ȳ)
    x̄ = ȳ[]
    return (NoTangent(), x̄)
  end
  return ITensor(x), ITensor_pullback
end

function rrule(::typeof(dag), x::ITensor)
  function dag_pullback(ȳ)
    x̄ = dag(unthunk(ȳ))
    return (NoTangent(), x̄)
  end
  return dag(x), dag_pullback
end

function rrule(::typeof(permute), x::ITensor, a...)
  y = permute(x, a...)
  function permute_pullback(ȳ)
    x̄ = permute(unthunk(ȳ), inds(x))
    ā = map_notangent(a)
    return (NoTangent(), x̄, ā...)
  end
  return y, permute_pullback
end

# Needed because by default it was calling the generic
# `rrule` for `tr` inside ChainRules.
# TODO: Raise an issue with ChainRules.
function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(tr), x::ITensor; kwargs...)
  return rrule_via_ad(config, ITensors._tr, x; kwargs...)
end

@non_differentiable combiner(::Indices)
