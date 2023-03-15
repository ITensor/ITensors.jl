export Combiner

# TODO: Have combiner store the locations
# of the uncombined and combined indices
# This can generalize to a Combiner that combines
# multiple set of indices, e.g. (i,j),(k,l) -> (a,b)
struct Combiner <: TensorStorage{Number}
  perm::Vector{Int}
  comb::Vector{Int}
  cind::Vector{Int}
  isconj::Bool
  function Combiner(perm::Vector{Int}, comb::Vector{Int}, cind::Vector{Int}, isconj::Bool)
    return new(perm, comb, cind, isconj)
  end
end

Combiner() = Combiner(Int[], Int[], Int[1], false)

Combiner(perm::Vector{Int}, comb::Vector{Int}) = Combiner(perm, comb, Int[1], false)

data(::Combiner) = NoData()
datatype(::Type{<:Combiner}) = NoData
setdata(C::Combiner, data::NoData) = C
blockperm(C::Combiner) = C.perm
blockcomb(C::Combiner) = C.comb
cinds(C::Combiner) = C.cind
isconj(C::Combiner) = C.isconj
setisconj(C::Combiner, isconj) = Combiner(blockperm(C), blockcomb(C), cinds(C), isconj)

function copy(C::Combiner)
  return Combiner(copy(blockperm(C)), copy(blockcomb(C)), copy(cinds(C)), isconj(C))
end

eltype(::Type{<:Combiner}) = Number

eltype(::Combiner) = eltype(Combiner)

promote_rule(::Type{<:Combiner}, StorageT::Type{<:Dense}) = StorageT

conj(::AllowAlias, C::Combiner) = setisconj(C, !isconj(C))
conj(::NeverAlias, C::Combiner) = conj(AllowAlias(), copy(C))

#
# CombinerTensor (Tensor using Combiner storage)
#

const CombinerTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:Combiner}

# The position of the combined index/dimension.
# By convention, it is the first one.
combinedind_position(combiner_tensor::CombinerTensor) = 1

function combinedind(combiner_tensor::CombinerTensor)
  return inds(combiner_tensor)[combinedind_position(combiner_tensor)]
end
# TODO: Rewrite in terms of `combinedind_position`.
function uncombinedinds(combiner_tensor::CombinerTensor)
  return deleteat(inds(combiner_tensor), combinedind_position(combiner_tensor))
end

function combinedind_label(combiner_tensor::CombinerTensor, combiner_tensor_labels)
  return combiner_tensor_labels[combinedind_position(combiner_tensor)]
end

function uncombinedind_labels(combiner_tensor::CombinerTensor, combiner_tensor_labels)
  return deleteat(combiner_tensor_labels, combinedind_position(combiner_tensor))
end

blockperm(C::CombinerTensor) = blockperm(storage(C))
blockcomb(C::CombinerTensor) = blockcomb(storage(C))

function is_index_replacement(
  tensor::Tensor, tensor_labels, combiner_tensor::CombinerTensor, combiner_tensor_labels
)
  return (ndims(combiner_tensor) == 2) &&
         isone(count(∈(tensor_labels), combiner_tensor_labels))
end

# Return if the combiner contraction is combining or uncombining.
# Check for valid contractions, for example when combining,
# only the combined index should be uncontracted, and when uncombining,
# only the combined index should be contracted.
function is_combining(
  tensor::Tensor, tensor_labels, combiner_tensor::CombinerTensor, combiner_tensor_labels
)
  is_combining = is_combining_no_check(
    tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
  check_valid_combiner_contraction(
    is_combining, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
  return is_combining
end

function is_combining_no_check(
  tensor::Tensor, tensor_labels, combiner_tensor::CombinerTensor, combiner_tensor_labels
)
  return combinedind_label(combiner_tensor, combiner_tensor_labels) ∉ tensor_labels
end

function check_valid_combiner_contraction(
  is_combining::Bool,
  tensor::Tensor,
  tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
)
  if !is_valid_combiner_contraction(
    is_combining, tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
  )
    return invalid_combiner_contraction_error(
      tensor, tensor_labels, combiner_tensor, combiner_tensor_labels
    )
  end
  return nothing
end

function is_valid_combiner_contraction(
  is_combining::Bool,
  tensor::Tensor,
  tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
)
  in_tensor_labels_op = is_combining ? ∉(tensor_labels) : ∈(tensor_labels)
  return isone(count(in_tensor_labels_op, combiner_tensor_labels))
end

function invalid_combiner_contraction_error(
  tensor::Tensor, tensor_labels, combiner_tensor::CombinerTensor, combiner_tensor_labels
)
  return error(
    """
    Trying to contract a tensor with indices:

    $(inds(tensor))

    and labels:

    $(tensor_labels)

    with a combiner tensor with indices:

    $(inds(combiner_tensor))

    and labels:

    $(combiner_tensor_labels).

    This is not a valid combiner contraction.

    If you are combining, the combined index of the combiner should be the only one uncontracted.

    If you are uncombining, the combined index of the combiner should be the only one contracted.

    By convention, the combined index should be the index in position $(combinedind_position(combiner_tensor)) of the combiner tensor.
    """,
  )
end

function show(io::IO, mime::MIME"text/plain", S::Combiner)
  println(io, "Permutation of blocks: ", S.perm)
  return println(io, "Combination of blocks: ", S.comb)
end

function show(io::IO, mime::MIME"text/plain", T::CombinerTensor)
  summary(io, T)
  println(io)
  return show(io, mime, storage(T))
end
