export Diag

struct Diag{T} <: TensorStorage
  data::Vector{T}
  Diag{T}(data::Vector) where {T} = new{T}(data)
  Diag{T}(size::Integer) where {T} = new{T}(Vector{T}(undef,size))
  Diag{T}(x::Number,size::Integer) where {T} = new{T}(fill(convert(T,x),size))
  Diag{T}() where {T} = new{T}(Vector{T}())
end

data(D::Diag) = D.data
# TODO: should this be the length of the diagonal
# or the product of the tensor dimensions?
length(D::Diag) = length(data(D))
eltype(D::Diag) = eltype(data(D))
getindex(D::Diag,i::Int) = data(D)[i]
*(D::Diag{T},x::S) where {T,S<:Number} = Dense{promote_type(T,S)}(x*data(D))
*(x::Number,D::Diag) = D*x

convert(::Type{Diag{T}},D::Diag) where T = Diag{T}(data(D))

# convert to Dense
storage_dense(D::Diag{T},is::IndexSet) where T = Dense{T}(vec(storage_convert(Array,D,is)))

# convert to complex
storage_complex(D::Diag{T}) where {T} = Diag{complex(T)}(complex(data(D)))

copy(D::Diag{T}) where {T} = Diag{T}(copy(data(D)))

# TODO: implement this
#outer(D1::Diag{T},D2::Diag{S}) where {T, S <:Number} = Diag{promote_type(T,S)}(vec(data(D1)*transpose(data(D2))))

# Convert to an array by setting the diagonal elements
function storage_convert(::Type{Array},D::Diag{T},is::IndexSet) where T
  A = zeros(T,dims(is))
  for i = 1:minDim(is)
    A[fill(i,length(is))...] = data(D)[i]
  end
  return A
end

storage_fill!(D::Diag,x::Number) = fill!(data(D),x)

# Get diagonal elements
# Gives zero for off-diagonal elements
function storage_getindex(Tstore::Diag{T},
                          Tis::IndexSet,
                          vals::Union{Int, AbstractVector{Int}}...) where {T}
  if all(y->y==vals[1],vals)
    return getindex(data(Tstore),vals[1])
  else
    return zero(T)
  end
end

# Set diagonal elements
# Throw error for off-diagonal
function storage_setindex!(Tstore::Diag,
                           Tis::IndexSet,
                           x::Union{<:Number, AbstractArray{<:Number}},
                           vals::Union{Int, AbstractVector{Int}}...)
  all(y->y==vals[1],vals) || error("Cannot set off-diagonal element of Diag storage")
  return setindex!(data(Tstore),x,vals[1])
end

function add!(Bstore::Diag,
              Bis::IndexSet,
              Astore::Diag,
              Ais::IndexSet,
              x::Number = 1)
  # This is just used to check if the index sets
  # are permutations of each other, maybe
  # this should be at the ITensor level
  p = calculate_permutation(Bis,Ais)
  Adata = data(Astore)
  Bdata = data(Bstore)
  if x == 1 
    Bdata .+= Adata
  else
    # TODO: is this just Bdata .+= x .* Adata?
    Bdata .= Bdata .+ x .* Adata
  end
end

function add!(Bstore::Dense,
              Bis::IndexSet,
              Astore::Diag,
              Ais::IndexSet,
              x::Number = 1)
  # This is just used to check if the index sets
  # are permutations of each other
  p = calculate_permutation(Bis,Ais)
  Adata = data(Astore)
  Bdata = data(Bstore)
  mindim = minDim(Bis)
  reshapeBdata = reshape(Bdata,dims(Bis))
  if x == 1
    for ii = 1:mindim
      # TODO: this should be optimized, maybe use
      # strides instead of reshape?
      reshapeBdata[fill(ii,order(Bis))...] += Adata[ii]
    end
  else
    for ii = 1:mindim
      reshapeBdata[fill(ii,order(Bis))...] += x*Adata[ii]
    end
  end
end

# Generic outer function that handles proper
# storage promotion
# TODO: should this handle promotion with storage
# type switching?
# TODO: we should combine all of the storage_add!
# outer wrappers into a single call that promotes
# based on the storage type, i.e. promote_type(Dense,Diag) -> Dense
function storage_add!(Bstore::Diag{BT},
                      Bis::IndexSet,
                      Astore::Diag{AT},
                      Ais::IndexSet,
                      x::Number = 1) where {BT,AT}
  NT = promote_type(AT,BT)
  if NT == BT
    add!(Bstore,Bis,Astore,Ais, x)
    return Bstore
  end
  Nstore = convert(Diag{NT},Bstore)
  add!(Nstore,Bis,Astore,Ais, x)
  return Nstore
end

function storage_add!(Bstore::Dense{BT},
                      Bis::IndexSet,
                      Astore::Diag{AT},
                      Ais::IndexSet,
                      x::Number = 1) where {BT,AT}
  NT = promote_type(AT,BT)
  if NT == BT
    add!(Bstore,Bis,Astore,Ais, x)
    return Bstore
  end
  Nstore = convert(Dense{NT},Bstore)
  add!(Nstore,Bis,Astore,Ais, x)
  return Nstore
end

function storage_add!(Bstore::Diag{BT},
                      Bis::IndexSet,
                      Astore::Dense{AT},
                      Ais::IndexSet,
                      x::Number = 1) where {BT,AT}
  Bstore = storage_dense(Bstore,Bis)
  NT = promote_type(AT,BT)
  if NT == BT
    add!(Bstore,Bis,Astore,Ais, x)
    return Bstore
  end
  Nstore = convert(Dense{NT},Bstore)
  add!(Nstore,Bis,Astore,Ais, x)
  return Nstore
end

# TODO: implement this
# This should not require any permutation
function storage_copyto!(Bstore::Diag,
                         Bis::IndexSet,
                         Astore::Diag,
                         Ais::IndexSet,
                         x::Number = 1)
  p = calculate_permutation(Bis,Ais)
  Adata = data(Astore)
  Bdata = data(Bstore)
  if is_trivial_permutation(p)
    if x == 1
      Bdata .= Adata
    else
      Bdata .= x .* Adata
    end
  else
    reshapeBdata = reshape(Bdata,dims(Bis))
    reshapeAdata = reshape(Adata,dims(Ais))
    if x == 1
      _add!(reshapeBdata,reshapeAdata,p,(a,b)->b)
    else
      _add!(reshapeBdata,reshapeAdata,p,(a,b)->x*b)
    end
  end
end

# TODO: Move to tensorstorage.jl?
function storage_mult!(Astore::Diag,
                       x::Number)
  Adata = data(Astore)
  rmul!(Adata, x)
end

# TODO: Move to tensorstorage.jl?
function storage_mult(Astore::Diag,
                      x::Number)
  Bstore = copy(Astore)
  storage_mult!(Bstore, x)
  return Bstore
end

# TODO: Move to tensorstorage.jl?
# For Real storage and complex scalar, promotion
# of the storage is needed
function storage_mult(Astore::Diag{T},
                      x::S) where {T<:Real,S<:Complex}
  Bstore = convert(Diag{promote_type(S,T)},Astore)
  storage_mult!(Bstore, x)
  return Bstore
end


# TODO: make this a special version of storage_add!()
# This shouldn't do anything to the data
function storage_permute!(Bstore::Diag,
                          Bis::IndexSet,
                          Astore::Diag,
                          Ais::IndexSet)
  p = calculate_permutation(Bis,Ais)
  Adata = data(Astore)
  Bdata = data(Bstore)
  if is_trivial_permutation(p)
    Bdata .= Adata
  else
    reshapeBdata = reshape(Bdata,dims(Bis))
    permutedims!(reshapeBdata,reshape(Adata,dims(Ais)),p)
  end
end

# TODO: move this to tensorstorage.jl?
function storage_dag(Astore::Diag,Ais::IndexSet)
  return dag(Ais),storage_conj(Astore)
end

# TODO: move this to tensorstorage.jl?
function storage_scalar(D::Diag)
  length(D)==1 && return D[1]
  throw(ErrorException("Cannot convert Diag -> Number for length of data greater than 1"))
end

# Contract function for Diag*Dense
function contract(Cinds::IndexSet,
                  Clabels::Vector{Int},
                  Astore::Diag{SA},
                  Ainds::IndexSet,
                  Alabels::Vector{Int},
                  Bstore::Dense{SB},
                  Binds::IndexSet,
                  Blabels::Vector{Int}) where {SA<:Number,SB<:Number}
  SC = promote_type(SA,SB)

  # Convert the arrays to a common type
  # since we will call BLAS
  Astore = convert(Diag{SC},Astore)
  Bstore = convert(Dense{SC},Bstore)

  Adims = dims(Ainds)
  Bdims = dims(Binds)
  Cdims = dims(Cinds)

  # Create storage for output tensor
  # This needs to be filled with zeros, since in general
  # the output of Diag*Dense will be sparse
  Cstore = Dense{SC}(0,prod(Cdims))

  # This seems unnecessary, and makes this function
  # non-generic. I think Dense should already store
  # the properly-shaped data (and be parametrized by the
  # order, and maybe store the size?)
  Adata = data(Astore)
  Bdata = reshape(data(Bstore),Bdims)
  Cdata = reshape(data(Cstore),Cdims)

  contract_diag_dense!(Cdata,Clabels,Adata,Alabels,Bdata,Blabels)
  return Cstore
end

# Contract function for Dense*Diag (calls Diag*Dense)
function contract(Cinds::IndexSet,
                  Clabels::Vector{Int},
                  Astore::Dense,
                  Ainds::IndexSet,
                  Alabels::Vector{Int},
                  Bstore::Diag,
                  Binds::IndexSet,
                  Blabels::Vector{Int})
  return contract(Cinds,Clabels,
                  Bstore,Binds,Blabels,
                  Astore,Ainds,Alabels)
end

# Contract function for Diag*Diag
function contract(Cinds::IndexSet,
                  Clabels::Vector{Int},
                  Astore::Diag{SA},
                  Ainds::IndexSet,
                  Alabels::Vector{Int},
                  Bstore::Diag{SB},
                  Binds::IndexSet,
                  Blabels::Vector{Int}) where {SA<:Number,SB<:Number}
  SC = promote_type(SA,SB)

  # Convert the arrays to a common type
  # since we will call BLAS
  Astore = convert(Diag{SC},Astore)
  Bstore = convert(Diag{SC},Bstore)
  Cstore = Diag{SC}(minDim(Cinds))

  Adata = data(Astore)
  Bdata = data(Bstore)
  Cdata = data(Cstore)

  contract_diag_diag!(Cdata,Clabels,Adata,Alabels,Bdata,Blabels)
  return Cstore
end

# This is generic, push it up to the ITensor level?
function contract_diag_dense!(Cdata::Array{T},Clabels::Vector{Int},
                              Adata::Vector{T},Alabels::Vector{Int},
                              Bdata::Array{T},Blabels::Vector{Int},
                              α::T=one(T),β::T=zero(T)) where T
  if(length(Alabels)==0)
    contract_scalar!(Cdata,Clabels,Bdata,Blabels,α*Adata[1],β)
  elseif(length(Blabels)==0)
    contract_scalar!(Cdata,Clabels,Adata,Alabels,α*Bdata[1],β)
  else
    _contract_diag_dense!(Cdata,Clabels,Adata,Alabels,Bdata,Blabels,α,β)
  end
  return
end

# This is generic, push it up to the ITensor level?
function contract_diag_diag!(Cdata::Vector{T},Clabels::Vector{Int},
                             Adata::Vector{T},Alabels::Vector{Int},
                             Bdata::Vector{T},Blabels::Vector{Int},
                             α::T=one(T),β::T=zero(T)) where T
  if(length(Alabels)==0)
    contract_scalar!(Cdata,Clabels,Bdata,Blabels,α*Adata[1],β)
  elseif(length(Blabels)==0)
    contract_scalar!(Cdata,Clabels,Adata,Alabels,α*Bdata[1],β)
  else
    _contract_diag_diag!(Cdata,Clabels,Adata,Alabels,Bdata,Blabels,α,β)
  end
  return
end

function _contract_diag_dense!(Cdata::Array{T,NC},Clabels::Vector{Int},
                               Adata::Vector{T},Alabels::Vector{Int},
                               Bdata::Array{T,NB},Blabels::Vector{Int},
                               α::T,β::T) where {T,NB,NC}
  if all(i -> i < 0, Blabels)  # If all of B is contracted
    dim = minimum(size(Bdata))
    if length(Clabels) == 0
      # all indices are summed over, just add the product of the diagonal
      # elements of A and B
      for i = 1:dim
        Cdata[1] += Adata[i]*Bdata[ntuple(_->i,Val(NB))...]
      end
    else
      # not all indices are summed over, set the diagonals of the result
      # to the product of the diagonals of A and B
      # TODO: should we make this return a Diag storage?
      for i = 1:dim
        Cdata[ntuple(_->i,Val(NC))...] = Adata[i]*Bdata[ntuple(_->i,Val(NB))...]
      end
    end
  else
    astarts = zeros(Int,length(Alabels))
    bstart = 0
    cstart = 0
    b_cstride = 0
    nbu = 0
    for ib = 1:length(Blabels)
      ia = findfirst(==(Blabels[ib]),Alabels)
      if(!isnothing(ia))
        b_cstride += stride(Bdata,ib)
        bstart += astarts[ia]*stride(Bdata,ib)
      else
        nbu += 1
      end
    end

    c_cstride = 0
    for ic = 1:length(Clabels)
      ia = findfirst(==(Clabels[ic]),Alabels)
      if(!isnothing(ia))
        c_cstride += stride(Cdata,ic)
        cstart += astarts[ia]*stride(Cdata,ic)
      end
    end

    # strides of the uncontracted dimensions of
    # Bdata
    bustride = zeros(Int,nbu)
    custride = zeros(Int,nbu)
    # size of the uncontracted dimensions of
    # Bdata, to be used in CartesianIndices
    busize = zeros(Int,nbu)
    n = 1
    for ib = 1:length(Blabels)
      if Blabels[ib] > 0
        bustride[n] = stride(Bdata,ib)
        busize[n] = size(Bdata,ib)
        ic = findfirst(==(Blabels[ib]),Clabels)
        custride[n] = stride(Cdata,ic)
        n += 1
      end
    end

    boffset_orig = 1-sum(strides(Bdata))
    coffset_orig = 1-sum(strides(Cdata))
    cartesian_inds = CartesianIndices(Tuple(busize))
    for inds in cartesian_inds
      boffset = boffset_orig
      coffset = coffset_orig
      for i in 1:nbu
        ii = inds[i]
        boffset += ii*bustride[i]
        coffset += ii*custride[i]
      end
      for j in 1:length(Adata)
        Cdata[cstart+j*c_cstride+coffset] += Adata[j]*Bdata[bstart+j*b_cstride+boffset]
      end
    end
  end
end

function _contract_diag_diag!(Cdata::Vector{T},Clabels::Vector{Int},
                              Adata::Vector{T},Alabels::Vector{Int},
                              Bdata::Vector{T},Blabels::Vector{Int},
                              α::T,β::T) where {T,NB,NC}
  if length(Clabels) == 0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    dim = length(Adata)  # == length(Bdata)
    for i = 1:dim
      Cdata[1] += Adata[i]*Bdata[i]
    end
  else
    dim = min(length(Adata),length(Bdata))
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    for i = 1:dim
      Cdata[i] = Adata[i]*Bdata[i]
    end
  end
end

## TODO: maybe we can do a special case for matrix, and
## otherwise turn it into an array
#function storage_svd(Astore::Diag{T},
#                     Lis::IndexSet,
#                     Ris::IndexSet;
#                     kwargs...) where {T}
#  maxdim::Int = get(kwargs,:maxdim,min(dim(Lis),dim(Ris)))
#  mindim::Int = get(kwargs,:mindim,1)
#  cutoff::Float64 = get(kwargs,:cutoff,0.0)
#  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
#  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
#  utags::String = get(kwargs,:utags,"Link,u")
#  vtags::String = get(kwargs,:vtags,"Link,v")
#  fastSVD::Bool = get(kwargs,:fastSVD,false)
#
#  if fastSVD
#    MU,MS,MV = svd(reshape(data(Astore),dim(Lis),dim(Ris)))
#  else
#    MU,MS,MV = recursiveSVD(reshape(data(Astore),dim(Lis),dim(Ris)))
#  end
#  MV = conj!(MV)
#
#  P = MS.^2
#  #@printf "  Truncating with maxdim=%d cutoff=%.3E\n" maxdim cutoff
#  truncate!(P;mindim=mindim,
#              maxdim=maxdim,
#              cutoff=cutoff,
#              absoluteCutoff=absoluteCutoff,
#              doRelCutoff=doRelCutoff)
#  dS = length(P)
#  if dS < length(MS)
#    MU = MU[:,1:dS]
#    resize!(MS,dS)
#    MV = MV[:,1:dS]
#  end
#
#  u = Index(dS,utags)
#  v = settags(u,vtags)
#  Uis,Ustore = IndexSet(Lis...,u),Diag{T}(vec(MU))
#  #TODO: make a diag storage
#  Sis,Sstore = IndexSet(u,v),Diag{Float64}(vec(Matrix(Diagonal(MS))))
#  Vis,Vstore = IndexSet(Ris...,v),Diag{T}(Vector{T}(vec(MV)))
#
#  return (Uis,Ustore,Sis,Sstore,Vis,Vstore)
#end
#
## TODO: maybe we can do a special case for matrix, and
## otherwise turn it into an array
#function storage_eigen(Astore::Diag{T},
#                       Lis::IndexSet,
#                       Ris::IndexSet;
#                       kwargs...) where {T}
#  maxdim::Int = get(kwargs,:maxdim,min(dim(Lis),dim(Ris)))
#  mindim::Int = get(kwargs,:mindim,1)
#  cutoff::Float64 = get(kwargs,:cutoff,0.0)
#  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
#  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
#  tags::TagSet = get(kwargs,:lefttags,"Link,u")
#  lefttags::TagSet = get(kwargs,:lefttags,tags)
#  righttags::TagSet = get(kwargs,:righttags,prime(lefttags))
#
#  dim_left = dim(Lis)
#  dim_right = dim(Ris)
#  MD,MU = eigen(Hermitian(reshape(data(Astore),dim_left,dim_right)))
#
#  # Sort by largest to smallest eigenvalues
#  p = sortperm(MD; rev = true)
#  MD = MD[p]
#  MU = MU[:,p]
#
#  #@printf "  Truncating with maxdim=%d cutoff=%.3E\n" maxdim cutoff
#  truncate!(MD;maxdim=maxdim,
#              cutoff=cutoff,
#              absoluteCutoff=absoluteCutoff,
#              doRelCutoff=doRelCutoff)
#  dD = length(MD)
#  if dD < size(MU,2)
#    MU = MU[:,1:dD]
#  end
#
#  #TODO: include truncation parameters as keyword arguments
#  u = Index(dD,lefttags)
#  v = settags(u,righttags)
#  Uis,Ustore = IndexSet(Lis...,u),Diag{T}(vec(MU))
#  #TODO: make a diag storage
#  Dis,Dstore = IndexSet(u,v),Diag{T}(vec(Matrix(Diagonal(MD))))
#  return (Uis,Ustore,Dis,Dstore)
#end
#
#function polar(A::Matrix)
#  U,S,V = svd(A) # calls LinearAlgebra.svd()
#  return U*V',V*Diagonal(S)*V'
#end
#
## TODO: maybe we can do a special case for matrix, and
## otherwise turn it into an array
#function storage_qr(Astore::Diag{T},
#                    Lis::IndexSet,
#                    Ris::IndexSet;
#                    kwargs...) where {T}
#  tags::TagSet = get(kwargs,:tags,"Link,u")
#  dim_left = dim(Lis)
#  dim_right = dim(Ris)
#  MQ,MP = qr(reshape(data(Astore),dim_left,dim_right))
#  dim_middle = min(dim_left,dim_right)
#  u = Index(dim_middle,tags)
#  #Must call Matrix() on MQ since the QR decomposition outputs a sparse
#  #form of the decomposition
#  Qis,Qstore = IndexSet(Lis...,u),Dense{T}(vec(Matrix(MQ)))
#  Pis,Pstore = IndexSet(u,Ris...),Dense{T}(vec(Matrix(MP)))
#  return (Qis,Qstore,Pis,Pstore)
#end
#
## TODO: maybe we can do a special case for matrix, and
## otherwise turn it into an array
#function storage_polar(Astore::Diag{T},
#                       Lis::IndexSet,
#                       Ris::IndexSet) where {T}
#  dim_left = dim(Lis)
#  dim_right = dim(Ris)
#  MQ,MP = polar(reshape(data(Astore),dim_left,dim_right))
#  dim_middle = min(dim_left,dim_right)
#  Uis = prime(Ris)
#  Qis,Qstore = IndexSet(Lis...,Uis...),Dense{T}(vec(MQ))
#  Pis,Pstore = IndexSet(Uis...,Ris...),Dense{T}(vec(MP))
#  return (Qis,Qstore,Pis,Pstore)
#end

