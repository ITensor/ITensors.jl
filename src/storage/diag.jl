export Diag

# Diag can have either Vector storage, in which case
# it is a general Diag tensor, or scalar storage,
# in which case it has a uniform value
mutable struct Diag{T} <: TensorStorage
  data::T
  Diag{T}(data::Vector) where {T<:AbstractVector} = new{T}(data)
  Diag{T}(size::Integer) where {T<:AbstractVector{S}} where S = new{T}(Vector{S}(undef,size))
  Diag{T}(x::Number,size::Integer) where {T<:AbstractVector} = new{T}(fill(convert(eltype(T),x),size))
  #Diag{T}() where {T} = new{T}(Vector{T}())
  # Make a uniform Diag storage
  Diag{T}(x::Number) where {T<:Number} = new{T}(convert(T,x))
  # Determine the storage type parameter from the input
  Diag(data::T) where T = new{T}(data)
end

data(D::Diag) = D.data
eltype(D::Diag) = eltype(data(D))
getindex(D::Diag{T},i::Int) where {T<:AbstractVector}= data(D)[i]
# Version of getindex for uniform (scalar) storage
getindex(D::Diag{T},i::Int) where {T<:Number} = data(D)
*(D::Diag{T},x::S) where {T<:AbstractVector,S<:Number} = Dense{promote_type(eltype(D),S)}(x*data(D))
*(x::Number,D::Diag) = D*x

#
# Type promotions involving Diag
# Useful for knowing how conversions should work when adding and contracting
#

Base.promote_type(::Type{Diag{T1}},::Type{Diag{T2}}) where {T1<:AbstractVector,T2<:AbstractVector} = Diag{promote_type(T1,T2)}

Base.promote_type(::Type{Diag{T1}},::Type{Diag{T2}}) where {T1<:Number,T2<:Number} = Diag{promote_type(T1,T2)}

Base.promote_type(::Type{Diag{T1}},::Type{Diag{T2}}) where {T1<:Number,T2<:Vector{S2}} where S2 = Diag{Vector{promote_type(T1,S2)}}

Base.promote_type(::Type{Diag{T1}},::Type{Diag{T2}}) where {T1<:Vector{S1},T2<:Number} where S1 = promote_type(Diag{T2},Diag{T1})

Base.promote_type(::Type{Dense{T1}},::Type{Diag{T2}}) where {T1<:Number,T2<:Vector{S2}} where S2 = Dense{promote_type(T1,S2)}

Base.promote_type(::Type{Dense{T1}},::Type{Diag{T2}}) where {T1<:Number,T2<:Number} = Dense{promote_type(T1,T2)}

Base.promote_type(::Type{Diag{T1}},::Type{Dense{T2}}) where {T1,T2} = promote_type(Dense{T2},Diag{T1})

# TODO: define length for Diag, should this be 
# the length of the diagonal
# or the product of the tensor dimensions?
# How is the length defined for scalar storage?
length(D::Diag{T}) where {T<:AbstractVector} = length(data(D))

# convert to Dense
function storage_dense(D::Diag,is::IndexSet)
  return Dense{eltype(D)}(vec(storage_convert(Array,D,is)))
end

# convert to complex
storage_complex(D::Diag) = Diag(complex(data(D)))

copy(D::Diag{T}) where {T} = Diag{T}(copy(data(D)))

# TODO: implement this in a sparse way
# For now, we will just make them dense since the output is dense anyway
function storage_outer(D1::Diag{T1},is1::IndexSet,D2::Diag{T2},is2::IndexSet) where {T1,T2}
  A1 = storage_dense(D1,is1)
  A2 = storage_dense(D2,is2)
  return storage_outer(A1,is1,A2,is2)
end

# Convert to an array by setting the diagonal elements
function storage_convert(::Type{Array},D::Diag,is::IndexSet)
  A = zeros(eltype(D),dims(is))
  for i = 1:minDim(is)
    A[fill(i,length(is))...] = D[i]
  end
  return A
end

Base.convert(::Type{Diag{T}},D::Diag) where T = Diag{T}(data(D))

function storage_convert(::Type{Dense{T}},D::Diag,is::IndexSet) where T
  return Dense{T}(vec(storage_convert(Array,D,is)))
end

storage_fill!(D::Diag,x::Number) = fill!(data(D),x)

function diag_getindex(Tstore::Diag{<:AbstractVector},
                       val::Int)
  return getindex(data(Tstore),val)
end

# Uniform case
function diag_getindex(Tstore::Diag{<:Number},
                       val::Int)
  return data(Tstore)
end

# Get diagonal elements
# Gives zero for off-diagonal elements
function storage_getindex(Tstore::Diag{T},
                          Tis::IndexSet,
                          vals::Union{Int, AbstractVector{Int}}...) where {T}
  if all(==(vals[1]),vals)
    return diag_getindex(Tstore,vals[1])
  else
    return zero(eltype(T))
  end
end

# Set diagonal elements
# Throw error for off-diagonal
function storage_setindex!(Tstore::Diag{<:AbstractVector},
                           Tis::IndexSet,
                           x::Union{<:Number, AbstractArray{<:Number}},
                           vals::Union{Int, AbstractVector{Int}}...)
  all(y->y==vals[1],vals) || error("Cannot set off-diagonal element of Diag storage")
  return setindex!(data(Tstore),x,vals[1])
end

function storage_setindex!(Tstore::Diag{<:Number},
                           Tis::IndexSet,
                           x::Union{<:Number, AbstractArray{<:Number}},
                           vals::Union{Int, AbstractVector{Int}}...)
  error("Cannot set elements of a uniform Diag storage")
end

# Add generic Diag's in-place
function _add!(Bstore::Diag,
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
    Bdata .+= x .* Adata
  end
end

# Uniform Diag case
function _add!(Bstore::Diag{BT},
               Bis::IndexSet,
               Astore::Diag{AT},
               Ais::IndexSet,
               x::Number = 1) where {BT<:Number,AT<:Number}
  # This is just used to check if the index sets
  # are permutations of each other, maybe
  # this should be at the ITensor level
  p = calculate_permutation(Bis,Ais)
  if x == 1
    Bstore.data += data(Astore)
  else
    Bstore.data += x * data(Astore)
  end
end

function _add!(Bstore::Dense,
               Bis::IndexSet,
               Astore::Diag,
               Ais::IndexSet,
               x::Number = 1)
  # This is just used to check if the index sets
  # are permutations of each other
  p = calculate_permutation(Bis,Ais)
  Bdata = data(Bstore)
  mindim = minDim(Bis)
  reshapeBdata = reshape(Bdata,dims(Bis))
  if x == 1
    for ii = 1:mindim
      # TODO: this should be optimized, maybe use
      # strides instead of reshape?
      reshapeBdata[fill(ii,order(Bis))...] += Astore[ii]
    end
  else
    for ii = 1:mindim
      reshapeBdata[fill(ii,order(Bis))...] += x*Astore[ii]
    end
  end
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
function storage_mult(Astore::Diag{T},
                      x::Number) where T
  Bdata = x*data(Astore)
  return Diag(Bdata)
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
function storage_scalar(D::Diag{T}) where {T<:AbstractVector}
  length(D)==1 && return D[1]
  throw(ErrorException("Cannot convert Diag -> Number for length of data greater than 1"))
end

function storage_scalar(D::Diag{T}) where {T<:Number}
  return data(T)
end

# Contract function for Diag*Dense
function _contract(Cinds::IndexSet,
                   Clabels::Vector{Int},
                   Astore::Diag{Vector{SA}},
                   Ainds::IndexSet,
                   Alabels::Vector{Int},
                   Bstore::Dense{SB},
                   Binds::IndexSet,
                   Blabels::Vector{Int}) where {SA<:Number,SB<:Number}
  SC = promote_type(SA,SB)

  # Convert the arrays to a common type
  # since we will call BLAS
  Astore = convert(Diag{Vector{SC}},Astore)
  Bstore = convert(Dense{SC},Bstore)

  Adims = dims(Ainds)
  Bdims = dims(Binds)
  Cdims = dims(Cinds)

  # Create storage for output tensor
  # This needs to be filled with zeros, since in general
  # the output of Diag*Dense will be sparse
  Cstore = Dense{SC}(zero(SC),prod(Cdims))

  # This seems unnecessary, and makes this function
  # non-generic. I think Dense should already store
  # the properly-shaped data (and be parametrized by the
  # order, and maybe store the size?)
  Adata = data(Astore)
  Bdata = reshape(data(Bstore),Bdims)
  Cdata = reshape(data(Cstore),Cdims)

  # Functions to do the contraction
  if(length(Alabels)==0)
    _contract_scalar!(Cdata,Clabels,Bdata,Blabels,Adata[1])
  elseif(length(Blabels)==0)
    _contract_scalar!(Cdata,Clabels,Adata,Alabels,Bdata[1])
  else
    _contract_diag_dense!(Cdata,Clabels,Adata,Alabels,Bdata,Blabels)
  end

  return Cstore
end

# Contract function for Diag uniform * Dense
function _contract(Cinds::IndexSet,
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
  #Astore = convert(Diag{SC},Astore)
  #Bstore = convert(Dense{SC},Bstore)

  Bdims = dims(Binds)
  Cdims = dims(Cinds)

  # Create storage for output tensor
  # This needs to be filled with zeros, since in general
  # the output of Diag*Dense will be sparse
  Cstore = Dense{SC}(zero(SC),prod(Cdims))

  # This seems unnecessary, and makes this function
  # non-generic. I think Dense should already store
  # the properly-shaped data (and be parametrized by the
  # order, and maybe store the size?)
  Adata = data(Astore)
  Bdata = reshape(data(Bstore),Bdims)
  Cdata = reshape(data(Cstore),Cdims)

  # Functions to do the contraction
  if(length(Alabels)==0)
    _contract_scalar!(Cdata,Clabels,Bdata,Blabels,Adata[1])
  elseif(length(Blabels)==0)
    _contract_scalar!(Cdata,Clabels,Adata,Alabels,Bdata[1])
  elseif(length(Alabels)==2 && length(Clabels)==length(Blabels))
    # This is just a replacement of the indices
    # TODO: This logic should be higher up
    for i = 1:length(Clabels)
      if Blabels[i] > 0
        Cinds[i] = Binds[i]
      else
        if Alabels[1] > 0
          Cinds[i] = Ainds[1]
        else
          Cinds[i] = Ainds[2]
        end
      end
    end
    Cdata .= Adata .* Bdata
  else
    _contract_diag_dense!(Cstore,Cinds,Clabels,
                          Astore,Ainds,Alabels,
                          Bstore,Binds,Blabels)
  end

  return Cstore
end

# Contract function for Dense*Diag (calls Diag*Dense)
function _contract(Cinds::IndexSet,
                   Clabels::Vector{Int},
                   Astore::Dense,
                   Ainds::IndexSet,
                   Alabels::Vector{Int},
                   Bstore::Diag,
                   Binds::IndexSet,
                   Blabels::Vector{Int})
  return _contract(Cinds,Clabels,
                   Bstore,Binds,Blabels,
                   Astore,Ainds,Alabels)
end

# Contract function for Diag*Diag
function _contract(Cinds::IndexSet,
                   Clabels::Vector{Int},
                   Astore::Diag{Vector{SA}},
                   Ainds::IndexSet,
                   Alabels::Vector{Int},
                   Bstore::Diag{Vector{SB}},
                   Binds::IndexSet,
                   Blabels::Vector{Int}) where {SA<:Number,SB<:Number}
  SC = promote_type(SA,SB)

  # Convert the arrays to a common type
  # since we will call BLAS
  Astore = convert(Diag{Vector{SC}},Astore)
  Bstore = convert(Diag{Vector{SC}},Bstore)
  Cstore = Diag{Vector{SC}}(minDim(Cinds))

  Adata = data(Astore)
  Bdata = data(Bstore)
  Cdata = data(Cstore)

  if(length(Alabels)==0)
    _contract_scalar!(Cdata,Clabels,Bdata,Blabels,Adata[1])
  elseif(length(Blabels)==0)
    _contract_scalar!(Cdata,Clabels,Adata,Alabels,Bdata[1])
  else
    _contract_diag_diag!(Cdata,Clabels,Adata,Alabels,Bdata,Blabels)
  end

  return Cstore
end

# Contract function for Diag uniform * Diag uniform
function _contract(Cinds::IndexSet,
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

  if length(Clabels) == 0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    min_dim = minimum(dims(Ainds)) # == length(Bdata)
    # Need to set to zero since
    # Cdata is originally uninitialized memory
    # (so causes random output)
    Cdata = zero(SC)
    for i = 1:min_dim
      Cdata += Adata*Bdata
    end
  else
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    Cdata = Adata*Bdata
  end

  Cstore.data = Cdata
  return Cstore
end

function _contract_diag_dense!(Cdata::Array{T,NC},Clabels::Vector{Int},
                               Adata::Vector{T},Alabels::Vector{Int},
                               Bdata::Array{T,NB},Blabels::Vector{Int}) where {T,NB,NC}
  if all(i -> i < 0, Blabels)  # If all of B is contracted
    min_dim = minimum(size(Bdata))
    if length(Clabels) == 0
      # all indices are summed over, just add the product of the diagonal
      # elements of A and B
      for i = 1:min_dim
        Cdata[1] += Adata[i]*Bdata[ntuple(_->i,Val(NB))...]
      end
    else
      # not all indices are summed over, set the diagonals of the result
      # to the product of the diagonals of A and B
      # TODO: should we make this return a Diag storage?
      for i = 1:min_dim
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

function _contract_diag_dense!(Cstore::Dense,Cinds,Clabels::Vector{Int},
                               Astore::Diag{TA},Ainds,Alabels::Vector{Int},
                               Bstore::Dense,Binds,Blabels::Vector{Int}) where {TA<:Number}
  if all(i -> i < 0, Blabels)  # If all of B is contracted
    Bdims = dims(Binds)
    min_dim = minimum(Bdims)
    rB = reshape(data(Bstore),Bdims)
    if length(Clabels) == 0
      # all indices are summed over, just add the product of the diagonal
      # elements of A and B
      # TODO: replace this with manual strides
      for i = 1:min_dim
        Cstore[1] += Astore[i]*rB[ntuple(_->i,length(Binds))...]
      end
    else
      # not all indices are summed over, set the diagonals of the result
      # to the product of the diagonals of A and B
      # TODO: should we make this return a Diag storage?
      for i = 1:min_dim
        Cstore[ntuple(_->i,length(Cinds))...] = Astore[i]*rB[ntuple(_->i,length(Binds))...]
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
        b_cstride += stride(Binds,ib)
        bstart += astarts[ia]*stride(Binds,ib)
      else
        nbu += 1
      end
    end

    c_cstride = 0
    for ic = 1:length(Clabels)
      ia = findfirst(==(Clabels[ic]),Alabels)
      if(!isnothing(ia))
        c_cstride += stride(Cinds,ic)
        cstart += astarts[ia]*stride(Cinds,ic)
      end
    end

    # strides of the uncontracted dimensions of
    # Bdata
    bustride = zeros(Int,nbu)
    custride = zeros(Int,nbu)
    # size of the uncontracted dimensions of
    # Bdata, to be used in CartesianIndices
    busize = zeros(Int,nbu)
    custride = zeros(Int,nbu)
    # size of the uncontracted dimensions of
    # Bdata, to be used in CartesianIndices
    busize = zeros(Int,nbu)
    n = 1
    for ib = 1:length(Blabels)
      if Blabels[ib] > 0
        bustride[n] = stride(Binds,ib)
        busize[n] = dim(Binds,ib) #size(Bstore,ib)
        ic = findfirst(==(Blabels[ib]),Clabels)
        custride[n] = stride(Cinds,ic)
        n += 1
      end
    end

    min_dim = minimum(dims(Binds))
    boffset_orig = 1-sum(strides(Binds))
    coffset_orig = 1-sum(strides(Cinds))
    cartesian_inds = CartesianIndices(Tuple(busize))
    for inds in cartesian_inds
      boffset = boffset_orig
      coffset = coffset_orig
      for i in 1:nbu
        ii = inds[i]
        boffset += ii*bustride[i]
        coffset += ii*custride[i]
      end
      for j in 1:min_dim
        Cstore[cstart+j*c_cstride+coffset] += Astore[j]*Bstore[bstart+j*b_cstride+boffset]
      end
    end
  end
end


# Maybe this works for uniform storage as well
function _contract_diag_diag!(Cdata::Vector{T},Clabels::Vector{Int},
                              Adata::Vector{T},Alabels::Vector{Int},
                              Bdata::Vector{T},Blabels::Vector{Int}) where T
  if length(Clabels) == 0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    Adim = length(Adata)  # == length(Bdata)
    # Need to set to zero since
    # Cdata is originally uninitialized memory
    # (so causes random output)
    Cdata[1] = zero(T)
    for i = 1:Adim
      Cdata[1] += Adata[i]*Bdata[i]
    end
  else
    min_dim = min(length(Adata),length(Bdata))
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    for i = 1:min_dim
      Cdata[i] = Adata[i]*Bdata[i]
    end
  end
end

