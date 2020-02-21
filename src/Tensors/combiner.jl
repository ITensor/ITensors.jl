export Combiner

# TODO: Have combiner store the locations
# of the uncombined and combined indices
# This can generalize to a Combiner that combines
# multiple set of indices, e.g. (i,j),(k,l) -> (a,b)
struct Combiner <: TensorStorage{Number}
  perm::Vector{Int}
  comb::Vector{Int}
  Combiner(perm::Vector{Int},comb::Vector{Int}) = new(perm,comb)
end

Combiner() = Combiner(Int[],Int[])

data(::Combiner) = error("Combiner storage has no data")

blockperm(C::Combiner) = C.perm
blockcomb(C::Combiner) = C.comb

Base.eltype(::Type{<:Combiner}) = Nothing
Base.eltype(::StoreT) where {StoreT<:Combiner} = eltype(StoreT)

Base.promote_rule(::Type{<:Combiner},StorageT::Type{<:Dense}) = StorageT

#
# CombinerTensor (Tensor using Combiner storage)
#

const CombinerTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:Combiner}

combinedindex(T::CombinerTensor) = inds(T)[1]
uncombinedinds(T::CombinerTensor) = popfirst(inds(T))

blockperm(C::CombinerTensor) = blockperm(store(C))
blockcomb(C::CombinerTensor) = blockcomb(store(C))

Base.conj(T::CombinerTensor) = T

function contraction_output(::TensorT1,
                            ::TensorT2,
                            indsR::IndsR) where {TensorT1<:CombinerTensor,
                                                 TensorT2<:DenseTensor,
                                                 IndsR}
  TensorR = contraction_output_type(TensorT1,TensorT2,IndsR)
  return similar(TensorR,indsR)
end

function contraction_output(T1::TensorT1,
                            T2::TensorT2,
                            indsR) where {TensorT1<:DenseTensor,
                                          TensorT2<:CombinerTensor}
  return contraction_output(T2,T1,indsR)
end

function contract!!(R::Tensor{<:Number,NR},
                    labelsR::NTuple{NR},
                    T1::CombinerTensor{<:Number,N1},
                    labelsT1::NTuple{N1},
                    T2::Tensor{<:Number,N2},
                    labelsT2::NTuple{N2}) where {NR,N1,N2}
  if N1 ≤ 1
    #println("identity")
    return R
  elseif N1 + N2 == NR
    error("Cannot perform outer product involving a combiner")
  elseif count_common(labelsT1,labelsT2) == 1 && N1 == 2
    # This is the case of index replacement
    ui = setdiff(labelsT1, labelsT2)[]
    newind = inds(T1)[findfirst(==(ui),labelsT1)]
    cpos1,cpos2 = intersect_positions(labelsT1,labelsT2)
    storeR = copy(store(T2))
    indsR = setindex(inds(T2),newind,cpos2)
    return Tensor(storeR,indsR)
  elseif count_common(labelsT1,labelsT2) == 1 && length(inds(T1)) != 2
    # This is the case of uncombining
    cpos1,cpos2 = intersect_positions(labelsT1,labelsT2)
    storeR = copy(store(T2))
    indsC = deleteat(inds(T1),cpos1)
    indsR = insertat(inds(T2),indsC,cpos2)
    return Tensor(storeR,indsR)
  elseif is_combiner(labelsT1,labelsT2)
    # This is the case of combining
    Alabels,Blabels = labelsT2,labelsT1
    final_labels    = contract_labels(Blabels, Alabels)
    final_labels_n  = contract_labels(labelsT1,labelsT2)
    indsR = inds(R)
    if final_labels != final_labels_n
      perm  = getperm(final_labels_n, final_labels)
      indsR = permute(inds(R), perm)
      labelsR = permute(labelsR, perm)
    end
    cpos1,cposR = intersect_positions(labelsT1,labelsR)
    labels_comb = deleteat(labelsT1,cpos1)
    vlR = [labelsR...]
    for (ii, li) in enumerate(labels_comb)
      insert!(vlR, cposR+ii, li)
    end
    deleteat!(vlR, cposR)
    labels_perm = tuple(vlR...) 
    perm = getperm(labels_perm,labelsT2)
    T2p = reshape(R,permute(inds(T2),perm))
    permutedims!(T2p,T2,perm)
    R = reshape(T2p,indsR)
  end
  return R
end

function contract!!(R::Tensor{<:Number,NR},
                    labelsR::NTuple{NR},
                    T1::Tensor{<:Number,N1},
                    labelsT1::NTuple{N1},
                    T2::CombinerTensor{<:Number,N2},
                    labelsT2::NTuple{N2}) where {NR,N1,N2}
  return contract!!(R,labelsR,T2,labelsT2,T1,labelsT1)
end

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::CombinerTensor)
  # Intentionally left blank
end

