
function ITensor(::Type{ElT},
                 flux::QN,
                 inds::IndexSet) where {ElT<:Number}
  blocks = nzblocks(flux,inds)
  T = BlockSparseTensor(blocks,inds)
  return itensor(T)
end

ITensor(::Type{T},
        flux::QN,
        inds::Index...) where {T<:Number} = ITensor(T,flux,IndexSet(inds...))

ITensor(flux::QN,inds::IndexSet) = ITensor(Float64,flux::QN,inds...)

ITensor(flux::QN,
        inds::Index...) = ITensor(flux,IndexSet(inds...))

function randomITensor(::Type{ElT},
                       flux::QN,
                       inds::IndexSet) where {ElT<:Number}
  T = ITensor(ElT,flux,inds)
  randn!(T)
  return T
end

function randomITensor(::Type{T},
                       flux::QN,
                       inds::Index...) where {T<:Number}
  return randomITensor(T,flux,IndexSet(inds...))
end

randomITensor(flux::QN,inds::IndexSet) = randomITensor(Float64,flux::QN,inds...)

randomITensor(flux::QN,
              inds::Index...) = randomITensor(flux,IndexSet(inds...))

Tensors.blockoffsets(T::ITensor) = blockoffsets(tensor(T))

Tensors.nnzblocks(T::ITensor) = nnzblocks(tensor(T))

flux(T::ITensor,block) = flux(inds(T),block)

function flux(T::ITensor)
  nnzblocks(T) == 0 && return QN()
  bofs = blockoffsets(T)
  block1 = block(bofs,1)
  return flux(T,block1)
end

#function combiner(inds::IndexSet; kwargs...)
#  tags = get(kwargs, :tags, "CMB,Link")
#  new_ind = Index(prod(dims(inds)), tags)
#  new_is = IndexSet(new_ind, inds)
#  return ITensor(Combiner(),new_is),new_ind
#end

function combiner(inds::QNIndex...; kwargs...)
  tags = get(kwargs, :tags, "CMB,Link")
  @show inds
  new_ind = ⊗(inds...)
  new_ind = settags(new_ind,tags)
  @show new_ind
  # Permute the blocks and combine them, outputing the 
  # new Index and the permutation
  new_ind,perm = combineqns(new_ind)
  @show new_ind
  @show perm
  return ITensor(Combiner(perm),IndexSet(new_ind,inds...)),new_ind
end

