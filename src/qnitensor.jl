
function ITensor(::Type{ElT},
                 flux::QN,
                 inds::IndexSet) where {ElT<:Number}
  blocks = nzblocks(flux,inds)
  T = BlockSparseTensor(ElT,blocks,inds)
  return itensor(T)
end

function ITensor(inds::QNIndex...)
  T = BlockSparseTensor(IndexSet(inds))
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

Tensors.nnz(T::ITensor) = nnz(tensor(T))

flux(T::ITensor,block) = flux(inds(T),block)

function flux(T::ITensor)
  nnzblocks(T) == 0 && return nothing
  bofs = blockoffsets(T)
  block1 = block(bofs,1)
  return flux(T,block1)
end

