
function contract(T::BlockSparseTensor{<:Number,NT},
                  labelsT,
                  C::CombinerTensor{<:Any,NC},
                  labelsC) where {NT,NC}
  # Get the label marking the combined index
  # By convention the combined index is the first one
  cpos_in_labelsC = 1
  clabel = labelsC[cpos_in_labelsC]
  if labelsC[1] ∉ labelsT
    c = combinedindex(C)
    labelsRc = contract_labels(labelsT,labelsC)
    indsRc = contract_inds(inds(T),labelsT,inds(C),labelsC,labelsRc)
    labels_uc = deleteat(labelsC,cpos_in_labelsC)
    cpos_in_labelsRc = findfirst(==(clabel),labelsRc)
    labelsR = insertat(labelsRc,labels_uc,cpos_in_labelsRc)
    perm = getperm(labelsR,labelsT)
    # TODO: should it be this instead?
    #perm = getperm(labelsT,labelsR)
    R = permutedims(T,perm)
    Rc = reshape(R,indsRc)
    return Rc
  else
    println("Uncombine: not implemented yet")
  end
end

#function contraction_output(::TensorT1,
#                            ::TensorT2,
#                            indsR::IndsR) where {TensorT1<:CombinerTensor,
#                                                 TensorT2<:BlockSparseTensor,
#                                                 IndsR}
#  TensorR = contraction_output_type(TensorT1,TensorT2,IndsR)
#  return similar(TensorR,indsR)
#end
#
#function contraction_output(T1::TensorT1,
#                            T2::TensorT2,
#                            indsR) where {TensorT1<:BlockSparseTensor,
#                                          TensorT2<:CombinerTensor}
#  return contraction_output(T2,T1,indsR)
#end

#function contract!!(R::BlockSparseTensor{<:Number,NR},
#                    labelsR::NTuple{NR},
#                    T1::CombinerTensor{<:Number,N1},
#                    labelsT1::NTuple{N1},
#                    T2::Tensor{<:Number,N2},
#                    labelsT2::NTuple{N2}) where {NR,N1,N2}
#  println("In contract!!(::BlockSparseTensor,...,CombinerTensor,...)")
#  #if N1 ≤ 1
#  #  #println("identity")
#  #  return R
#  #elseif N1 + N2 == NR
#  #  error("Cannot perform outer product involving a combiner")
#  #elseif count_common(labelsT1,labelsT2) == 1 && N1 == 2
#  #  # This is the case of index replacement
#  #  ui = setdiff(labelsT1, labelsT2)[]
#  #  newind = inds(T1)[findfirst(==(ui),labelsT1)]
#  #  cpos1,cpos2 = intersect_positions(labelsT1,labelsT2)
#  #  storeR = copy(store(T2))
#  #  indsR = setindex(inds(T2),newind,cpos2)
#  #  return Tensor(storeR,indsR)
#  #elseif count_common(labelsT1,labelsT2) == 1 && length(inds(T1)) != 2
#  #  # This is the case of uncombining
#  #  cpos1,cpos2 = intersect_positions(labelsT1,labelsT2)
#  #  storeR = copy(store(T2))
#  #  indsC = deleteat(inds(T1),cpos1)
#  #  indsR = insertat(inds(T2),indsC,cpos2)
#  #  return Tensor(storeR,indsR)
#  #elseif is_combiner(labelsT1,labelsT2)
#  #  # This is the case of combining
#  #  Alabels,Blabels = labelsT2,labelsT1
#  #  final_labels    = contract_labels(Blabels, Alabels)
#  #  final_labels_n  = contract_labels(labelsT1,labelsT2)
#  #  indsR = inds(R)
#  #  if final_labels != final_labels_n
#  #    perm  = getperm(final_labels_n, final_labels)
#  #    indsR = permute(inds(R), perm)
#  #    labelsR = permute(labelsR, perm)
#  #  end
#  #  cpos1,cposR = intersect_positions(labelsT1,labelsR)
#  #  labels_comb = deleteat(labelsT1,cpos1)
#  #  vlR = [labelsR...]
#  #  for (ii, li) in enumerate(labels_comb)
#  #    insert!(vlR, cposR+ii, li)
#  #  end
#  #  deleteat!(vlR, cposR)
#  #  labels_perm = tuple(vlR...) 
#  #  perm = getperm(labels_perm,labelsT2)
#  #  T2p = reshape(R,permute(inds(T2),perm))
#  #  permutedims!(T2p,T2,perm)
#  #  R = reshape(T2p,indsR)
#  #end
#  #return R
#end

#function contract!!(R::Tensor{<:Number,NR},
#                    labelsR::NTuple{NR},
#                    T1::Tensor{<:Number,N1},
#                    labelsT1::NTuple{N1},
#                    T2::CombinerTensor{<:Number,N2},
#                    labelsT2::NTuple{N2}) where {NR,N1,N2}
#  return contract!!(R,labelsR,T2,labelsT2,T1,labelsT1)
#end

