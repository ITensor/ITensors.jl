export Diag

struct Diag{T} <: TensorStorage
  data::Vector{T}
  Diag{T}(data::Vector) where {T} = new{T}(convert(Vector{T},data))
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

convert(::Type{Diag{T}},D::Diag) where {T} = Diag{T}(data(D))

# convert to complex
storage_complex(D::Diag{T}) where {T} = Diag{complex(T)}(complex(data(D)))

copy(D::Diag{T}) where {T} = Diag{T}(copy(data(D)))

# TODO: implement this
#outer(D1::Diag{T},D2::Diag{S}) where {T, S <:Number} = Diag{promote_type(T,S)}(vec(data(D1)*transpose(data(D2))))

# TODO: convert to an array by setting the diagonal elements
function storage_convert(::Type{Array},D::Diag,is::IndexSet)
  A = zeros(eltype(D),dims(is))
  for i = 1:minDim(is)
    A[fill(i,length(is))...] = data(D)[i]
  end
  return A
end

storage_fill!(D::Diag,x::Number) = fill!(data(D),x)

# TODO: only get diagonal elements
# Should it give zero for off-diagonal elements?
function storage_getindex(Tstore::Diag{T},
                          Tis::IndexSet,
                          vals::Union{Int, AbstractVector{Int}}...) where {T}
  if all(y->y==vals[1],vals)
    return getindex(data(Tstore),vals[1])
  else
    return zero(T)
  end
end

# TODO: only set diagonal elements
# Throw error for off-diagonal
function storage_setindex!(Tstore::Diag,
                           Tis::IndexSet,
                           x::Union{<:Number, AbstractArray{<:Number}},
                           vals::Union{Int, AbstractVector{Int}}...)
  all(y->y==vals[1],vals) || error("Cannot set off-diagonal element of Diag storage")
  return setindex!(data(Tstore),x,vals[1])
end

# TODO: implement this
# This should not require any permutation
#function add!(Bstore::Diag,
#              Bis::IndexSet,
#              Astore::Diag,
#              Ais::IndexSet,
#              x::Number = 1)
#  p = calculate_permutation(Bis,Ais)
#  Adata = data(Astore)
#  Bdata = data(Bstore)
#  if is_trivial_permutation(p)
#    if x == 1
#      Bdata .+= Adata
#    else
#      Bdata .= Bdata .+ x .* Adata
#    end
#  else
#    reshapeBdata = reshape(Bdata,dims(Bis))
#    reshapeAdata = reshape(Adata,dims(Ais))
#    if x == 1
#      _add!(reshapeBdata,reshapeAdata,p)
#    else
#      _add!(reshapeBdata,reshapeAdata,p,(a,b)->a+x*b)
#    end
#  end
#end

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

