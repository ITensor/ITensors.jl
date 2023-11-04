## function contract(
##   t_src::Tensor{T,N,<:BlockSparseArray{T,N}},
##   labels_src,
##   t_comb::Tensor{Any,M,<:CombinerArray{M}},
##   labels_comb,
## ) where {T,N,M}
##   array_dest = contract(storage(t_src), labels_src, storage(t_comb), labels_comb)
##   inds_dest = if is_combining(storage(t_src), labels_src, storage(t_comb), labels_comb)
##     contract_inds_combine(t_src, labels_src, t_comb, labels_comb)
##   else
##     contract_inds_uncombine(t_src, labels_src, t_comb, labels_comb)
##   end
##   return tensor(array_dest, inds_dest)
## end
## 
## function contract(
##   t_comb::Tensor{Any,M,<:CombinerArray{M}},
##   labels_comb,
##   t_src::Tensor{T,N,<:BlockSparseArray{T,N}},
##   labels_src,
## ) where {T,N,M}
##   return contract(t_src, labels_src, t_comb, labels_comb)
## end
## 
## function contract_inds_combine(
##   t_src::Tensor{<:Any,<:Any,<:BlockSparseArray},
##   labels_src,
##   t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
##   labels_comb,
## )
##   labels_dest = contract_labels(labels_comb, labels_src)
##   return contract_inds(inds(t_comb), labels_comb, inds(t_src), labels_src, labels_dest)
## end
## 
## function contract_inds_uncombine(
##   t_src::Tensor{<:Any,<:Any,<:BlockSparseArray},
##   labels_src,
##   t_comb::Tensor{<:Any,<:Any,<:CombinerArray},
##   labels_comb,
## )
##   inds_dest, = contract_inds_uncombine(inds(t_src), labels_src, inds(t_comb), labels_comb)
##   return inds_dest
## end
