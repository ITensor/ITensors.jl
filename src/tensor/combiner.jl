export Combiner

struct Combiner <: TensorStorage
end

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

function count_unique(labelsT1,labelsT2)
  count = 0
  for l1 ∈ labelsT1
    l1 ∉ labelsT2 && (count += 1)
  end
  return count
end

function count_common(labelsT1,labelsT2)
  count = 0
  for l1 ∈ labelsT1
    l1 ∈ labelsT2 && (count += 1)
  end
  return count
end

function intersect_positions(labelsT1,labelsT2)
  for i1 = 1:length(labelsT1)
    for i2 = 1:length(labelsT2)
      if labelsT1[i1] == labelsT2[i2]
        return i1,i2
      end
    end
  end
  return nothing
end

function is_replacement(labelsT1,labelsT2)
  return count_unique(labelsT1,labelsT2) == 1 &&
         count_common(labelsT1,labelsT2) == 1
end

function is_combiner(labelsT1,labelsT2)
  return count_unique(labelsT1,labelsT2) == 1 &&
         count_common(labelsT1,labelsT2) > 1
end

function is_uncombiner(labelsT1,labelsT2)
  return count_unique(labelsT1,labelsT2) > 1 &&
         count_common(labelsT1,labelsT2) == 1
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

#function storage_contract(CSstore::CombinerStorage,
#                          Cis::IndexSet,
#                          Dstore::Dense,
#                          dis::IndexSet)
#  cind = Cis[1]
#  if hasindex(dis, cind) # has combined index, uncombine
#    cpos = indexposition(dis,cind)
#    dinds = inds(dis)
#    Cinds = inds(Cis)
#    Nis = IndexSet(vcat(dinds[1:cpos-1], Cinds[2:end], dinds[cpos+1:end]))
#    return Nis, Dstore
#  else # lacks combined index, combine
#    # dis doesn't have cind, replace
#    # Cis[1], Cis[2], ... with cind, may need to permute
#    j_1_pos = indexposition(dis,Cis[2])
#    if isnothing(j_1_pos)
#      throw(ArgumentError("tensor missing index $(Cis[2]) in combiner-tensor product"))
#    end
#
#    # Check if Cis[2], Cis[3], ... are grouped together (contiguous)
#    # and in same order as on combiner
#    contig_sameord = true
#    j = j_1_pos+1
#    for c=3:length(Cis)
#      if j > length(dis) || (dis[j] != Cis[c])
#        contig_sameord = false
#        break
#      end
#      j += 1
#    end
#
#    if contig_sameord
#      lend = j_1_pos-1
#      rstart = j_1_pos+length(Cis)-1
#      dinds = inds(dis)
#      Nis = IndexSet(vcat(dinds[1:lend], cind, dinds[rstart:end]))
#    else # permutation required
#      # Set P destination values to -1 to mark indices that need to be assigned destinations:
#      P = fill(-1, length(dis))
#      # permute combined indices to the front, in same order as in Cis:
#      ni = 1
#      for c in 2:length(Cis)
#        j = indexposition(dis, Cis[c])
#        if isnothing(j)
#          throw(ArgumentError("tensor missing index $(Cis[c]) in combiner-tensor product"))
#        end
#        P[j] = ni
#        ni += 1
#      end
#
#      Nis = IndexSet(length(dis)+2-length(Cis))
#      Nis[1] = cind
#      i = 2
#      for j in 1:length(dis)
#        if P[j] == -1
#          P[j] = ni
#          ni += 1
#          Nis[i] = dis[j]
#          i += 1
#        end
#      end
#
#      ddata = vec(permutedims(reshape(data(Dstore), dims(dis)), invperm(P)))
#      Dstore = Dense{eltype(Dstore)}(ddata)
#    end
#    return Nis, Dstore
#  end
#end
#
#function storage_contract(Dstore::Dense, 
#                          dis::IndexSet, 
#                          CSstore::CombinerStorage, 
#                          Cis::IndexSet) 
#  return storage_contract(CSstore, Cis, Dstore, dis)
#end

