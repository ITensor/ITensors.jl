#<fermions>:
before_combiner_signs(T, labelsT, indsT, C, labelsC, indsC, labelsR, indsR) = T
after_combiner_signs(R, labelsR, indsR, C, labelsC, indsC) = R

function contract(T::BlockSparseTensor, labelsT, C::CombinerTensor, labelsC)
  #@timeit_debug timer "Block sparse (un)combiner" begin
  # Get the label marking the combined index
  # By convention the combined index is the first one
  # TODO: consider storing the location of the combined
  # index in preperation for multiplce combined indices
  cpos_in_labelsC = 1
  clabel = labelsC[cpos_in_labelsC]
  c = combinedindex(C)
  labels_uc = deleteat(labelsC, cpos_in_labelsC)
  if labelsC[1] ∉ labelsT
    # Combine
    labelsRc = contract_labels(labelsC, labelsT)
    cpos_in_labelsRc = findfirst(==(clabel), labelsRc)
    labelsRuc = insertat(labelsRc, labels_uc, cpos_in_labelsRc)
    indsRc = contract_inds(inds(C), labelsC, inds(T), labelsT, labelsRc)

    #<fermions>:
    T = before_combiner_signs(T, labelsT, inds(T), C, labelsC, inds(C), labelsRc, indsRc)

    perm = getperm(labelsRuc, labelsT)
    ucpos_in_labelsT = Tuple(findall(x -> x in labels_uc, labelsT))
    Rc = permutedims_combine(T, indsRc, perm, ucpos_in_labelsT, blockperm(C), blockcomb(C))
    return Rc
  else
    # Uncombine
    labelsRc = labelsT
    cpos_in_labelsRc = findfirst(==(clabel), labelsRc)
    # Move combined index to first position
    if cpos_in_labelsRc != 1
      labelsRc_orig = labelsRc
      labelsRc = deleteat(labelsRc, cpos_in_labelsRc)
      labelsRc = insertafter(labelsRc, clabel, 0)
      cpos_in_labelsRc = 1
      perm = getperm(labelsRc, labelsRc_orig)
      T = permutedims(T, perm)
      labelsT = permute(labelsT, perm)
    end
    labelsRuc = insertat(labelsRc, labels_uc, cpos_in_labelsRc)
    indsRuc = contract_inds(inds(C), labelsC, inds(T), labelsT, labelsRuc)

    # <fermions>:
    T = before_combiner_signs(T, labelsT, inds(T), C, labelsC, inds(C), labelsRuc, indsRuc)

    Ruc = uncombine(T, indsRuc, cpos_in_labelsRc, blockperm(C), blockcomb(C))

    # <fermions>:
    Ruc = after_combiner_signs(Ruc, labelsRuc, indsRuc, C, labelsC, inds(C))

    return Ruc
  end
  #end # @timeit
end

function contract(C::CombinerTensor, labelsC, T::BlockSparseTensor, labelsT)
  return contract(T, labelsT, C, labelsC)
end

# Special case when no indices are combined
# XXX: no copy
contract(T::BlockSparseTensor, labelsT, C::CombinerTensor{<:Any,0}, labelsC) = copy(T)
