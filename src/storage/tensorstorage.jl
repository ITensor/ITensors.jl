export data

abstract type TensorStorage end

#
# Generic ITensor storage functions
#

storage_randn!(S::TensorStorage) = randn!(data(S))
storage_norm(S::TensorStorage) = norm(data(S))
storage_conj(S::T) where {T<:TensorStorage}= T(conj(data(S)))

#
# This high level part of contract is generic to different
# storage types
#
# TODO: make this storage_contract!(), where C is pre-allocated.
#       This will allow for in-place multiplication
# TODO: optimize the contraction logic so C doesn't get permuted?
#
function storage_contract(Astore::TensorStorage,
                          Ais::IndexSet,
                          Bstore::TensorStorage,
                          Bis::IndexSet)
  if length(Ais)==0
    Cis = Bis
    Cstore = storage_scalar(Astore)*Bstore
  elseif length(Bis)==0
    Cis = Ais
    Cstore = storage_scalar(Bstore)*Astore
  else
    #TODO: check for special case when Ais and Bis are disjoint sets
    #I think we should do this analysis outside of storage_contract, at the ITensor level
    #(since it is universal for any storage type and just analyzes in indices)
    (Alabels,Blabels) = compute_contraction_labels(Ais,Bis)
    if is_outer(Alabels,Blabels)
      Cis = IndexSet(Ais,Bis)
      Cstore = outer(Astore,Bstore)
    else
      (Cis,Clabels) = contract_inds(Ais,Alabels,Bis,Blabels)
      Cstore = contract(Cis,Clabels,Astore,Ais,Alabels,Bstore,Bis,Blabels)
    end
  end
  return (Cis,Cstore)
end

