function permutedims(tensor::Tensor, perm)
  (ndims(tensor) == length(perm) && isperm(perm)) ||
    throw(ArgumentError("no valid permutation of dimensions"))
  output_tensor = NDTensors.similar(tensor, permute(inds(tensor), perm))
  return permutedims!!(output_tensor, tensor, perm)
end

# Version that may overwrite the result or allocate
# and return the result of the permutation.
# Similar to `BangBang.jl` notation:
# https://juliafolds.github.io/BangBang.jl/stable/.
function permutedims!!(output_tensor::Tensor, tensor::Tensor, perm, f::Function=(r, t) -> t)
  Base.checkdims_perm(output_tensor, tensor, perm)
  permutedims!(output_tensor, tensor, perm, f)
  return output_tensor
end

function permutedims!(output_tensor::Tensor, tensor::Tensor, perm, f::Function=(r, t) -> t)
  Base.checkdims_perm(output_tensor, tensor, perm)
  error(
    "`perutedims!(output_tensor::Tensor, tensor::Tensor, perm, f::Function=(r, t) -> t)` not implemented for `typeof(output_tensor) = $(typeof(output_tensor))`, `typeof(tensor) = $(typeof(tensor))`, `perm = $perm`, and `f = $f`.",
  )
  return output_tensor
end

function (x::Number * tensor::Tensor)
  return NDTensors.tensor(x * storage(tensor), inds(tensor))
end
(tensor::Tensor * x::Number) = x * tensor

function (tensor::Tensor / x::Number)
  return NDTensors.tensor(storage(tensor) / x, inds(tensor))
end

function contraction_output_type(
  tensortype1::Type{<:Tensor}, tensortype2::Type{<:Tensor}, inds
)
  return similartype(promote_type(tensortype1, tensortype2), inds)
end

function contraction_output(
  tensor1::Tensor, labelstensor1, tensor2::Tensor, labelstensor2, labelsoutput_tensor
)
  indsoutput_tensor = contract_inds(
    inds(tensor1), labelstensor1, inds(tensor2), labelstensor2, labelsoutput_tensor
  )
  output_tensor = contraction_output(tensor1, tensor2, indsoutput_tensor)
  return output_tensor
end

# Trait returning true if the two tensors or storage types can
# contract with each other.
@traitdef CanContract{X,Y}
#! format: off
@traitimpl CanContract{X,Y} <- can_contract(X, Y)
#! format: on

# Assume storage types can contract with each other
can_contract(tensor1::Type, tensor2::Type) = true
function can_contract(tensor1::Type{<:Tensor}, tensor2::Type{<:Tensor})
  return can_contract(storagetype(tensor1), storagetype(tensor2))
end

function can_contract(tensor1::TensorStorage, tensor2::TensorStorage)
  return can_contract(typeof(tensor1), typeof(tensor2))
end
function can_contract(tensor1::Tensor, tensor2::Tensor)
  return can_contract(typeof(tensor1), typeof(tensor2))
end

# Version where output labels aren't supplied
@traitfn function contract(
  tensor1::TensorT1, labels_tensor1, tensor2::TensorT2, labels_tensor2
) where {TensorT1<:Tensor,TensorT2<:Tensor;CanContract{TensorT1,TensorT2}}
  labelsoutput_tensor = contract_labels(labels_tensor1, labels_tensor2)
  return contract(tensor1, labels_tensor1, tensor2, labels_tensor2, labelsoutput_tensor)
end

@traitfn function contract(
  tensor1::TensorT1, labels_tensor1, tensor2::TensorT2, labels_tensor2
) where {TensorT1<:Tensor,TensorT2<:Tensor;!CanContract{TensorT1,TensorT2}}
  return error(
    "Can't contract tensor of storage type $(storagetype(tensor1)) with tensor of storage type $(storagetype(tensor2)).",
  )
end

function contract(
  tensor1::Tensor, labelstensor1, tensor2::Tensor, labelstensor2, labelsoutput_tensor
)
  # TODO: put the contract_inds logic into contraction_output,
  # call like output_tensor = contraction_ouput(tensor1,labelstensor1,tensor2,labelstensor2)
  #indsoutput_tensor = contract_inds(inds(tensor1),labelstensor1,inds(tensor2),labelstensor2,labelsoutput_tensor)
  output_tensor = contraction_output(
    tensor1, labelstensor1, tensor2, labelstensor2, labelsoutput_tensor
  )
  # contract!! version here since the output output_tensor may not
  # be mutable (like UniformDiag)
  output_tensor = contract!!(
    output_tensor, labelsoutput_tensor, tensor1, labelstensor1, tensor2, labelstensor2
  )
  return output_tensor
end

# Overload this function for immutable storage types
function _contract!!(
  output_tensor::Tensor,
  labelsoutput_tensor,
  tensor1::Tensor,
  labelstensor1,
  tensor2::Tensor,
  labelstensor2,
  α::Number=1,
  β::Number=0,
)
  if α ≠ 1 || β ≠ 0
    contract!(
      output_tensor,
      labelsoutput_tensor,
      tensor1,
      labelstensor1,
      tensor2,
      labelstensor2,
      α,
      β,
    )
  else
    contract!(
      output_tensor, labelsoutput_tensor, tensor1, labelstensor1, tensor2, labelstensor2
    )
  end
  return output_tensor
end

# Is this generic for all storage types?
function contract!!(
  output_tensor::Tensor,
  labelsoutput_tensor,
  tensor1::Tensor,
  labelstensor1,
  tensor2::Tensor,
  labelstensor2,
  α::Number=1,
  β::Number=0,
)
  Noutput_tensor = ndims(output_tensor)
  N1 = ndims(tensor1)
  N2 = ndims(tensor2)
  if (N1 ≠ 0) && (N2 ≠ 0) && (N1 + N2 == Noutput_tensor)
    # Outer product
    (α ≠ 1 || β ≠ 0) && error(
      "contract!! not yet implemented for outer product tensor contraction with non-trivial α and β",
    )
    # TODO: permute tensor1 and tensor2 appropriately first (can be more efficient
    # then permuting the result of tensor1⊗tensor2)
    # TODO: implement the in-place version directly
    output_tensor = outer!!(output_tensor, tensor1, tensor2)
    labelsoutput_tensorp = (labelstensor1..., labelstensor2...)
    perm = getperm(labelsoutput_tensor, labelsoutput_tensorp)
    if !is_trivial_permutation(perm)
      output_tensorp = reshape(output_tensor, (inds(tensor1)..., inds(tensor2)...))
      output_tensor = permutedims!!(output_tensor, copy(output_tensorp), perm)
    end
  else
    if α ≠ 1 || β ≠ 0
      output_tensor = _contract!!(
        output_tensor,
        labelsoutput_tensor,
        tensor1,
        labelstensor1,
        tensor2,
        labelstensor2,
        α,
        β,
      )
    else
      output_tensor = _contract!!(
        output_tensor, labelsoutput_tensor, tensor1, labelstensor1, tensor2, labelstensor2
      )
    end
  end
  return output_tensor
end

function outer!!(output_tensor::Tensor, tensor1::Tensor, tensor2::Tensor)
  outer!(output_tensor, tensor1, tensor2)
  return output_tensor
end

function outer end

const ⊗ = outer
