
#function ITensor(::Type{T},
#                 inds::IndexSet{N}) where {T<:Number,N}
#  return ITensor{N}(Dense{float(T)}(zeros(float(T),dim(inds))),inds)
#end

function flux(inds::IndexSet,block)
  qntot = QN()
  for i in 1:ndims(inds)
    qntot += qn(inds[i],block[i])
  end
  return qntot
end

function nzblocks(qn::QN,inds::IndexSet{N}) where {N}
  blocks = NTuple{N,Int}[]
  for block in eachblock(inds)
    if flux(inds,block) == qn
      push!(blocks,Tuple(block))
    end
  end
  return blocks
end

function ITensor(::Type{ElT},
                 flux::QN,
                 inds::IndexSet) where {ElT<:Number}
  @show flux
  @show inds
  blocks = nzblocks(flux,inds)
  @show blocks
  T = BlockSparseTensor(blocks,inds)
  @show T
  #ITensor(T,IndexSet(inds...))
end

ITensor(::Type{T},
        flux::QN,
        inds::Index...) where {T<:Number} = ITensor(T,flux,IndexSet(inds...))

ITensor(flux::QN,inds::IndexSet) = ITensor(Float64,flux::QN,inds...)

ITensor(flux::QN,
        inds::Index...) where {T<:Number} = ITensor(flux,IndexSet(inds...))

