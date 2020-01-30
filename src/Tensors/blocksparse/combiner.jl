
function contract(T::BlockSparseTensor{<:Number,NT},
                  labelsT,
                  C::CombinerTensor{<:Any,NC},
                  labelsC) where {NT,NC}
  # Get the label marking the combined index
  # By convention the combined index is the first one
  # TODO: consider storing the location of the combined
  # index in preperation for multiplce combined indices
  cpos_in_labelsC = 1
  clabel = labelsC[cpos_in_labelsC]
  c = combinedindex(C)
  labels_uc = deleteat(labelsC,cpos_in_labelsC)
  if labelsC[1] ∉ labelsT
    labelsRc = contract_labels(labelsT,labelsC)
    cpos_in_labelsRc = findfirst(==(clabel),labelsRc)
    labelsRuc = insertat(labelsRc,labels_uc,cpos_in_labelsRc)
    indsRc = contract_inds(inds(T),labelsT,inds(C),labelsC,labelsRc)
    perm = getperm(labelsRuc,labelsT)
    ucpos_in_labelsT = Tuple(findall(x->x in labels_uc,labelsT))
    Rc = permutedims_combine(T,indsRc,perm,ucpos_in_labelsT,blockperm(C),blockcomb(C))
    return Rc
  else
    labelsRc = labelsT
    cpos_in_labelsRc = findfirst(==(clabel),labelsRc)
    labelsRuc = insertat(labelsRc,labels_uc,cpos_in_labelsRc)
    indsRuc = contract_inds(inds(T),labelsT,inds(C),labelsC,labelsRuc)
    Ruc = uncombine(T,indsRuc,cpos_in_labelsRc,blockperm(C),blockcomb(C))
    return Ruc
  end
end

