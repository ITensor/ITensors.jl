
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

flux(T::ITensor,block) = flux(inds(T),block)

function flux(T::ITensor)
  nnzblocks(T) == 0 && return QN()
  bofs = blockoffsets(T)
  block1 = block(bofs,1)
  return flux(T,block1)
end

function combiner(inds::QNIndex...; kwargs...)
  # TODO: support combining multiple set of indices
  tags = get(kwargs, :tags, "CMB,Link")
  do_combineblocks = get(kwargs, :combineblocks, false)
  new_ind = ⊗(inds...)
  if all(i->dir(i)!=Out,inds)
    new_ind = dag(new_ind)
    new_ind = replaceqns(new_ind,-qnblocks(new_ind))
  end
  new_ind = settags(new_ind,tags)
  if do_combineblocks
    println("Combining blocks in combiner is not implemented yet")
    # TODO: sort and combine the blocks
    # Use sortperm to get the permutation, and store
    # which blocks are combined in comb
    # For example, QN(1),QN(0),QN(2),QN(1):
    # perm = (2,1,4,3)
    # comb = (1,2,2,3) # Part of block 1,2, or 3
    # The permutation is passed to the combiner
    comb_ind,perm,comb = combineblocks(new_ind)
    return ITensor(Combiner(perm,comb,new_ind),IndexSet(comb_ind,dag.(inds)...)),comb_ind
  end
  return ITensor(Combiner(),IndexSet(new_ind,dag.(inds)...)),new_ind
end

