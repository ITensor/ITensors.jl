is_cu(::Type{<:Array}) = false
is_cu(::Type{<:CuArray}) = true

is_cu(X::Type{<:TensorStorage}) = is_cu(NDTensors.datatype(X))
is_cu(X::Type{<:Tensor}) = is_cu(NDTensors.storagetype(X))
is_cu(::Type{ITensor}) = error("Unknown")

# Special cases
is_cu(X::Type{<:Combiner}) = false

is_cu(x::CuArray) = is_cu(typeof(x))
is_cu(x::Array) = is_cu(typeof(x))
is_cu(x::TensorStorage) = is_cu(typeof(x))
is_cu(x::Tensor) = is_cu(typeof(x))
is_cu(x::ITensor) = is_cu(tensor(x))

mixed_cu_cpu(T1::Type, T2::Type{<:Combiner}) = false
mixed_cu_cpu(T1::Type{<:Combiner}, T2::Type) = mixed_cu_cpu(T2, T1)
mixed_cu_cpu(T1::Type, T2::Type) = (is_cu(T1) ⊻ is_cu(T2))

@traitdef MixedCuCPU{T1,T2}
@traitimpl MixedCuCPU{T1,T2} < -mixed_cu_cpu(T1, T2)

@traitfn function can_contract(
  ::Type{T1}, ::Type{T2}
) where {T1<:TensorStorage,T2<:TensorStorage;!MixedCuCPU{T1,T2}}
  return true
end
@traitfn function can_contract(
  ::Type{T1}, ::Type{T2}
) where {T1<:TensorStorage,T2<:TensorStorage;MixedCuCPU{T1,T2}}
  return false
end
