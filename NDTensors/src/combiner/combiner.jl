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

combinedindex(T::CombinerTensor) = inds(T)[1]
uncombinedinds(T::CombinerTensor) = popfirst(inds(T))

blockperm(C::CombinerTensor) = blockperm(storage(C))
blockcomb(C::CombinerTensor) = blockcomb(storage(C))

function contraction_output(
  ::TensorT1, ::TensorT2, indsR::IndsR
) where {TensorT1<:CombinerTensor,TensorT2<:DenseTensor,IndsR}
  TensorR = contraction_output_type(TensorT1, TensorT2, IndsR)
  return similar(TensorR, indsR)
end

function contraction_output(
  T1::TensorT1, T2::TensorT2, indsR
) where {TensorT1<:DenseTensor,TensorT2<:CombinerTensor}
  return contraction_output(T2, T1, indsR)
end

function contract!!(R::Tensor, labelsR, T1::CombinerTensor, labelsT1, T2::Tensor, labelsT2)
  NR = ndims(R)
  N1 = ndims(T1)
  N2 = ndims(T2)
  if N1 ≤ 1
    # Empty combiner, acts as multiplying by 1
    R = permutedims!!(R, T2, getperm(labelsR, labelsT2))
    return R
  elseif N1 + N2 == NR
    error("Cannot perform outer product involving a combiner")
  elseif count_common(labelsT1, labelsT2) == 1 && N1 == 2
    # This is the case of index replacement
    ui = setdiff(labelsT1, labelsT2)[]
    newind = inds(T1)[findfirst(==(ui), labelsT1)]
    cpos1, cpos2 = intersect_positions(labelsT1, labelsT2)
    storeR = copy(storage(T2))
    indsR = setindex(inds(T2), newind, cpos2)
    return tensor(storeR, indsR)
  elseif count_common(labelsT1, labelsT2) == 1 && length(inds(T1)) != 2
    # This is the case of uncombining
    cpos1, cpos2 = intersect_positions(labelsT1, labelsT2)
    storeR = copy(storage(T2))
    indsC = deleteat(inds(T1), cpos1)
    indsR = insertat(inds(T2), indsC, cpos2)
    return tensor(storeR, indsR)
  elseif is_combiner(labelsT1, labelsT2)
    # This is the case of combining
    Alabels, Blabels = labelsT2, labelsT1
    final_labels = contract_labels(Blabels, Alabels)
    final_labels_n = contract_labels(labelsT1, labelsT2)
    indsR = inds(R)
    if final_labels != final_labels_n
      perm = getperm(final_labels_n, final_labels)
      indsR = permute(inds(R), perm)
      labelsR = permute(labelsR, perm)
    end
    cpos1, cposR = intersect_positions(labelsT1, labelsR)
    labels_comb = deleteat(labelsT1, cpos1)
    vlR = [labelsR...]
    for (ii, li) in enumerate(labels_comb)
      insert!(vlR, cposR + ii, li)
    end
    deleteat!(vlR, cposR)
    labels_perm = tuple(vlR...)
    perm = getperm(labels_perm, labelsT2)
    T2p = reshape(R, permute(inds(T2), perm))
    permutedims!(T2p, T2, perm)
    R = reshape(T2p, indsR)
  end
  return R
end

function contract!!(R::Tensor, labelsR, T1::Tensor, labelsT1, T2::CombinerTensor, labelsT2)
  return contract!!(R, labelsR, T2, labelsT2, T1, labelsT1)
end

function contract(T1::DiagTensor, labelsT1, T2::CombinerTensor, labelsT2)
  return contract(dense(T1), labelsT1, T2, labelsT2)
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
