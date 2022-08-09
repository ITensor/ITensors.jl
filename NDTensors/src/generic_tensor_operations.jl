function permutedims(T::Tensor{<:Number,N}, perm::NTuple{N,Int}) where {N}
  Tp = similar(T, permute(inds(T), perm))
  Tp = permutedims!!(Tp, T, perm)
  return Tp
end

function permutedims(::Tensor, ::Tuple{Vararg{Int}})
  return error("Permutation size must match tensor order")
end

function (x::Number * T::Tensor)
  return tensor(x * storage(T), inds(T))
end
(T::Tensor * x::Number) = x * T

function (T::Tensor / x::Number)
  return tensor(storage(T) / x, inds(T))
end

function contraction_output_type(
  ::Type{TensorT1}, ::Type{TensorT2}, ::Type{IndsR}
) where {TensorT1<:Tensor,TensorT2<:Tensor,IndsR}
  return similartype(promote_type(TensorT1, TensorT2), IndsR)
end

function contraction_output(T1::Tensor, labelsT1, T2::Tensor, labelsT2, labelsR)
  indsR = contract_inds(inds(T1), labelsT1, inds(T2), labelsT2, labelsR)
  R = contraction_output(T1, T2, indsR)
  return R
end

# Trait returning true if the two tensors or storage types can
# contract with each other.
@traitdef CanContract{X,Y}
@traitimpl CanContract{X,Y} <- can_contract(X, Y)

# Assume storage types can contract with each other
can_contract(T1::Type, T2::Type) = true
function can_contract(T1::Type{<:Tensor}, T2::Type{<:Tensor})
  return can_contract(storagetype(T1), storagetype(T2))
end

can_contract(t1::TensorStorage, t2::TensorStorage) = can_contract(typeof(t1), typeof(t2))
can_contract(t1::Tensor, t2::Tensor) = can_contract(typeof(t1), typeof(t2))

contract_labels(labels_t1, labels_t2) = tuple(symdiff(labels_t1, labels_t2)...)

# Version where output labels aren't supplied
@traitfn function contract(
  t1::T1, labels_t1, t2::T2, labels_t2
) where {T1<:Tensor,T2<:Tensor;CanContract{T1,T2}}
  labels_R = contract_labels(labels_t1, labels_t2)
  return contract(t1, labels_t1, t2, labels_t2, labels_R)
end

@traitfn function contract(
  t1::T1, labels_t1, t2::T2, labels_t2
) where {T1<:Tensor,T2<:Tensor;!CanContract{T1,T2}}
  return error(
    "Can't contract tensor of storage type $(storagetype(t1)) with tensor of storage type $(storagetype(t2)).",
  )
end

function contract(T1::Tensor, labelsT1, T2::Tensor, labelsT2, labelsR)
  R = contraction_output(T1, T2, labelsR)
  return contract!!(R, labelsR, T1, labelsT1, T2, labelsT2)
end

# Overload this function for immutable storage types
function contract!!(
  R::Tensor, labelsR, T1::Tensor, labelsT1, T2::Tensor, labelsT2
)
  contract!(R, labelsR, T1, labelsT1, T2, labelsT2)
  return R
end
