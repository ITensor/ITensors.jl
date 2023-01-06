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
#! format: off
@traitimpl CanContract{X,Y} <- can_contract(X, Y)
#! format: on

# Assume storage types can contract with each other
can_contract(T1::Type, T2::Type) = true
function can_contract(T1::Type{<:Tensor}, T2::Type{<:Tensor})
  return can_contract(storagetype(T1), storagetype(T2))
end

can_contract(t1::TensorStorage, t2::TensorStorage) = can_contract(typeof(t1), typeof(t2))
can_contract(t1::Tensor, t2::Tensor) = can_contract(typeof(t1), typeof(t2))

# Version where output labels aren't supplied
@traitfn function contract(
  t1::T1, labels_t1, t2::T2, labels_t2
) where {T1<:Tensor,T2<:Tensor;CanContract{T1,T2}}
  labelsR = contract_labels(labels_t1, labels_t2)
  return contract(t1, labels_t1, t2, labels_t2, labelsR)
end

@traitfn function contract(
  t1::T1, labels_t1, t2::T2, labels_t2
) where {T1<:Tensor,T2<:Tensor;!CanContract{T1,T2}}
  return error(
    "Can't contract tensor of storage type $(storagetype(t1)) with tensor of storage type $(storagetype(t2)).",
  )
end

function contract(T1::Tensor, labelsT1, T2::Tensor, labelsT2, labelsR)
  # TODO: put the contract_inds logic into contraction_output,
  # call like R = contraction_ouput(T1,labelsT1,T2,labelsT2)
  #indsR = contract_inds(inds(T1),labelsT1,inds(T2),labelsT2,labelsR)
  R = contraction_output(T1, labelsT1, T2, labelsT2, labelsR)
  # contract!! version here since the output R may not
  # be mutable (like UniformDiag)
  R = contract!!(R, labelsR, T1, labelsT1, T2, labelsT2)
  return R
end

# Overload this function for immutable storage types
function _contract!!(
  R::Tensor, labelsR, T1::Tensor, labelsT1, T2::Tensor, labelsT2, α::Number=1, β::Number=0
)
  if α ≠ 1 || β ≠ 0
    contract!(R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
  else
    contract!(R, labelsR, T1, labelsT1, T2, labelsT2)
  end
  return R
end

# Is this generic for all storage types?
function contract!!(
  R::Tensor, labelsR, T1::Tensor, labelsT1, T2::Tensor, labelsT2, α::Number=1, β::Number=0
)
  NR = ndims(R)
  N1 = ndims(T1)
  N2 = ndims(T2)
  if (N1 ≠ 0) && (N2 ≠ 0) && (N1 + N2 == NR)
    # Outer product
    (α ≠ 1 || β ≠ 0) && error(
      "contract!! not yet implemented for outer product tensor contraction with non-trivial α and β",
    )
    # TODO: permute T1 and T2 appropriately first (can be more efficient
    # then permuting the result of T1⊗T2)
    # TODO: implement the in-place version directly
    R = outer!!(R, T1, T2)
    labelsRp = (labelsT1..., labelsT2...)
    perm = getperm(labelsR, labelsRp)
    if !is_trivial_permutation(perm)
      Rp = reshape(R, (inds(T1)..., inds(T2)...))
      R = permutedims!!(R, copy(Rp), perm)
    end
  else
    if α ≠ 1 || β ≠ 0
      R = _contract!!(R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
    else
      R = _contract!!(R, labelsR, T1, labelsT1, T2, labelsT2)
    end
  end
  return R
end

function outer!!(R::Tensor, T1::Tensor, T2::Tensor)
  outer!(R, T1, T2)
  return R
end

function outer end

const ⊗ = outer
