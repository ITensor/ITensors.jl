
function contract(T::BlockSparseTensor,
                  labelsT,
                  C::CombinerTensor,
                  labelsC)
  # Get the label marking the combined index
  # By convention the combined index is the first one
  # TODO: consider storing the location of the combined
  # index in preperation for multiplce combined indices
  cpos_in_labelsC = 1
  clabel = labelsC[cpos_in_labelsC]
  c = combinedindex(C)
  labels_uc = deleteat(labelsC,cpos_in_labelsC)
  if labelsC[1] ∉ labelsT
    # Combine
    labelsRc = contract_labels(labelsC,labelsT)
    cpos_in_labelsRc = findfirst(==(clabel),labelsRc)
    labelsRuc = insertat(labelsRc,labels_uc,cpos_in_labelsRc)
    indsRc = contract_inds(inds(C),labelsC,inds(T),labelsT,labelsRc)
    perm = getperm(labelsRuc,labelsT)
    ucpos_in_labelsT = Tuple(findall(x->x in labels_uc,labelsT))

    #Rc = permutedims_combine(T,indsRc,perm,ucpos_in_labelsT,blockperm(C),blockcomb(C))

    Rc = permutedims_combine_2(T,indsRc,perm,ucpos_in_labelsT,blockperm(C),blockcomb(C))

    return Rc
  else
    # Uncombine
    labelsRc = labelsT
    cpos_in_labelsRc = findfirst(==(clabel),labelsRc)

    #@show cpos_in_labelsRc

    # Move combined index to first position
    if cpos_in_labelsRc != 1
      labelsRc_orig = labelsRc
      labelsRc = deleteat(labelsRc,cpos_in_labelsRc)
      labelsRc = insertafter(labelsRc,clabel,0)
      cpos_in_labelsRc = 1

      #@show clabel
      #@show labelsRc_orig
      #@show labelsRc

      perm = getperm(labelsRc,labelsRc_orig)

      #@show perm

      T = permutedims(T,perm)
      labelsT = permute(labelsT,perm)
    end

    #@show labelsT
    #@show inds(T)

    labelsRuc = insertat(labelsRc,labels_uc,cpos_in_labelsRc)

    #@show labelsRuc

    indsRuc = contract_inds(inds(C),labelsC,inds(T),labelsT,labelsRuc)

    #@show indsRuc

    Ruc = uncombine_2(T,indsRuc,cpos_in_labelsRc,blockperm(C),blockcomb(C))

    #@show Ruc
    #@show uncombine_2(T,indsRuc,cpos_in_labelsRc,blockperm(C),blockcomb(C))

    return Ruc
  end
end

contract(C::CombinerTensor,
         labelsC,
         T::BlockSparseTensor,
         labelsT) = contract(T,labelsT,C,labelsC)

# Special case when no indices are combined
contract(T::BlockSparseTensor,
         labelsT,
         C::CombinerTensor{<:Any,0},
         labelsC) = copy(T)

