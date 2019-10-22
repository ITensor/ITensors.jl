export Combiner

struct Combiner <: TensorStorage{Number}
end

data(::Combiner) = error("Combiner storage has no data")

Base.eltype(::Type{<:Combiner}) = Nothing
Base.eltype(::StoreT) where {StoreT<:Combiner} = eltype(StoreT)

Base.promote_rule(::Type{<:Combiner},StorageT::Type{<:Dense}) = StorageT

#
# CombinerTensor (Tensor using Dense storage)
#

const CombinerTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:Combiner}

function contraction_output_type(TensorT1::Type{<:CombinerTensor},
                                 TensorT2::Type{<:DenseTensor},
                                 indsR)
  return similar_type(promote_type(TensorT1,TensorT2),indsR)
end

function contraction_output_type(TensorT1::Type{<:DenseTensor},
                                 TensorT2::Type{<:CombinerTensor},
                                 indsR)
  return contraction_output_type(TensorT2,TensorT1,indsR)
end

function contraction_output(TensorT1::Type{<:CombinerTensor},
                            TensorT2::Type{<:DenseTensor},
                            indsR)
  return similar(contraction_output_type(TensorT1,TensorT2,indsR),indsR)
end

function contraction_output(TensorT1::Type{<:DenseTensor},
                            TensorT2::Type{<:CombinerTensor},
                            indsR)
  return contraction_output(TensorT2,TensorT1,indsR)
end

function contract!!(R::Tensor{<:Number,NR},
                    labelsR::NTuple{NR},
                    T1::CombinerTensor{<:Any,N1},
                    labelsT1::NTuple{N1},
                    T2::Tensor{<:Number,N2},
                    labelsT2::NTuple{N2}) where {NR,N1,N2}
  if N1 ≤ 1
    println("identity")
    return R
  elseif N1 + N2 == NR
    error("Cannot perform outer product involving a combiner")
  elseif count_common(labelsT1,labelsT2) == 1
    # This is the case of Index replacement or
    # uncombining
    # TODO: handle the case where inds(R) and inds(T1)
    # are not ordered the same?
    # Could just use a permutedims...
    return Tensor(store(T2),inds(R))
  elseif is_combiner(labelsT1,labelsT2)
    # This is the case of combining
    cpos1,cposR = intersect_positions(labelsT1,labelsR)
    labels_comb = deleteat(labelsT1,cpos1)
    labels_perm = insertat(labelsR,labels_comb,cposR)
    perm = getperm(labels_perm,labelsT2)
    T2p = reshape(R,permute(inds(T2),perm))
    permutedims!(T2p,T2,perm)
    R = reshape(T2p,inds(R))
  end
  return R
end

function contract!!(R::Tensor{<:Number,NR},
                    labelsR::NTuple{NR},
                    T1::Tensor{<:Number,N1},
                    labelsT1::NTuple{N1},
                    T2::CombinerTensor{<:Any,N2},
                    labelsT2::NTuple{N2}) where {NR,N1,N2}
  return contract!!(R,labelsR,T2,labelsT2,T1,labelsT1)
end

