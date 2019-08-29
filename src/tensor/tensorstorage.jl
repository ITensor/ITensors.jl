export data

abstract type TensorStorage end

#
# Generic ITensor storage functions
#

#storage_randn!(S::TensorStorage) = randn!(data(S))
#storage_norm(S::TensorStorage) = norm(data(S))
#storage_conj(S::T) where {T<:TensorStorage}= T(conj(data(S)))

#
# This high level part of contract is generic to different
# storage types
#
# TODO: make this storage_contract!(), where C is pre-allocated.
#       This will allow for in-place multiplication
# TODO: optimize the contraction logic so C doesn't get permuted?
#
#function storage_contract(Astore::TensorStorage,
#                          Ais::IndexSet,
#                          Bstore::TensorStorage,
#                          Bis::IndexSet)
#  if length(Ais)==0
#    Cis = Bis
#    Cstore = storage_scalar(Astore)*Bstore
#  elseif length(Bis)==0
#    Cis = Ais
#    Cstore = storage_scalar(Bstore)*Astore
#  else
#    #TODO: check for special case when Ais and Bis are disjoint sets
#    #I think we should do this analysis outside of storage_contract, at the ITensor level
#    #(since it is universal for any storage type and just analyzes in indices)
#    (Alabels,Blabels) = compute_contraction_labels(Ais,Bis)
#    if is_outer(Alabels,Blabels)
#      Cstore,Cis = storage_outer(Astore,Ais,Bstore,Bis)
#    else
#      (Cis,Clabels) = contract_inds(Ais,Alabels,Bis,Blabels)
#      Cstore = _contract(Cis,Clabels,Astore,Ais,Alabels,Bstore,Bis,Blabels)
#    end
#  end
#  return (Cis,Cstore)
#end

# Generic outer function that handles proper
# storage promotion
# TODO: should this handle promotion with storage
# type switching?
# TODO: we should combine all of the storage_add!
# outer wrappers into a single call that promotes
# based on the storage type, i.e. promote_type(Dense,Diag) -> Dense
#function storage_add!(Bstore::BT,
#                      Bis::IndexSet,
#                      Astore::AT,
#                      Ais::IndexSet,
#                      x::Number = 1) where {BT<:TensorStorage,AT<:TensorStorage}
#  NT = promote_type(AT,BT)
#  if NT == BT
#    _add!(Bstore,Bis,Astore,Ais, x)
#    return Bstore
#  end
#  Nstore = storage_convert(NT,Bstore,Bis)
#  _add!(Nstore,Bis,Astore,Ais, x)
#  return Nstore
#end

