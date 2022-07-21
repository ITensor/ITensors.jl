# Trait type indicating the object is either on CPU
# or on a CUDA device (for example a type that doesn't
# have any data, like a Combiner or uniform Diagonal
# tensor).
struct CPUorCUDA end

is_cu(::Type{<:Number}) = CPUorCUDA()
is_cu(::Type{NDTensors.NoData}) = CPUorCUDA()
is_cu(::Type{<:Array}) = false
is_cu(::Type{<:CuArray}) = true

# Handle Array wrappers like `ReshapedArray`.
@traitfn function is_cu(arraytype::Type{T}) where {T; IsWrappedArray{T}}
  return is_cu(parenttype(arraytype))
end

is_cu(X::Type{<:TensorStorage}) = is_cu(NDTensors.datatype(X))
is_cu(X::Type{<:Tensor}) = is_cu(NDTensors.storagetype(X))
is_cu(::Type{ITensor}) = error("Unknown")

is_cu(x::CuArray) = is_cu(typeof(x))
is_cu(x::Array) = is_cu(typeof(x))

is_cu(x::TensorStorage) = is_cu(typeof(x))
is_cu(x::Tensor) = is_cu(typeof(x))
is_cu(x::ITensor) = is_cu(typeof(tensor(x)))
is_cu(x::MPS) = all(is_cu, x)
is_cu(x::MPO) = all(is_cu, x)

mixed_cu_cpu(::Bool, ::CPUorCUDA) = false
mixed_cu_cpu(::CPUorCUDA, ::Bool) = false
mixed_cu_cpu(::CPUorCUDA, ::CPUorCUDA) = false
mixed_cu_cpu(is_cu1::Bool, is_cu2::Bool) = (is_cu1 ⊻ is_cu2)
mixed_cu_cpu(T1::Type, T2::Type) = mixed_cu_cpu(is_cu(T1), is_cu(T2))

@traitdef MixedCuCPU{T1,T2}

#! format: off
@traitimpl MixedCuCPU{T1,T2} <- mixed_cu_cpu(T1, T2)
#! format: on

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
