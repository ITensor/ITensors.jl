
struct Dense{T} <: TensorStorage
  data::Vector{T}
  Dense{T}(data::Vector{T}) where {T} = new{T}(data)
  Dense{T}(size::Integer) where {T} = new{T}(zeros(size))
  Dense{T}(x::Number,size::Integer) where {T} = new{T}(fill(x,size))
  Dense{T}() where {T} = new{T}(Vector{T}())
end

data(D::Dense) = D.data
length(D::Dense) = length(data(D))
eltype(D::Dense) = eltype(data(D))

copy(D::Dense{T}) where {T} = Dense{T}(copy(data(D)))

storage_convert(::Type{Array},D::Dense,is::IndexSet) = reshape(data(D),dims(is))

function storage_getindex(Tstore::Dense,
                          Tis::IndexSet,
                          vals::Int...)
  return getindex(reshape(data(Tstore),dims(Tis)),vals...)
end

function storage_setindex!(Tstore::Dense,Tis::IndexSet,x::Number,vals::Int...)
  return setindex!(reshape(data(Tstore),dims(Tis)),x,vals...)
end

# TODO: optimize this permutation (this does an extra unnecassary permutation
# since permutedims!() doesn't give the option to add the permutation to the original array)
# Maybe wrap the c version?
function storage_add!(Bstore::Dense,Bis::IndexSet,Astore::Dense,Ais::IndexSet)
  p = calculate_permutation(Bis,Ais)
  Adata = data(Astore)
  Bdata = data(Bstore)
  if is_trivial_permutation(p)
    Bdata .+= Adata
  else
    reshapeBdata = reshape(Bdata,dims(Bis))
    permAdata = permutedims(reshape(Adata,dims(Ais)),p)
    reshapeBdata .+= permAdata
  end
end

# TODO: make this a special version of storage_add!()
# Make sure the permutation is optimized
function storage_permute!(Bstore::Dense,Bis::IndexSet,Astore::Dense,Ais::IndexSet)
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

function storage_dag(Astore::Dense,Ais::IndexSet)
  return dag(Ais),storage_conj(Astore)
end

function storage_scalar(D::Dense)
  if length(D)==1
    return data(D)[1]
  else
    throw(ErrorException("Cannot convert Dense -> Number for length of data greater than 1"))
  end
end

# TODO: make this storage_contract!(), where C is pre-allocated. 
#       This will allow for in-place multiplication
# TODO: optimize the contraction logic so C doesn't get permuted?
function storage_contract(Astore::TensorStorage,
                          Ais::IndexSet,
                          Bstore::TensorStorage,
                          Bis::IndexSet)
  (Alabels,Blabels) = compute_contraction_labels(Ais,Bis)
  (Cis,Clabels) = contract_inds(Ais,Alabels,Bis,Blabels)
  Cstore = contract(Cis,Clabels,Astore,Ais,Alabels,Bstore,Bis,Blabels)
  return (Cis,Cstore)
end

function storage_svd(Astore::Dense{T},
                     Lis::IndexSet,
                     Ris::IndexSet;
                     maxm::Int=min(dim(Lis),dim(Ris)),
                     minm::Int=1,
                     cutoff::Float64=0.0,
                     absoluteCutoff::Bool=false,
                     doRelCutoff::Bool=true,
                     utags::String="Link,u",
                     vtags::String="Link,v"
                    ) where {T}
  MU,MS,MV = svd(reshape(data(Astore),dim(Lis),dim(Ris)))

  sqr(x) = x^2
  P = sqr.(MS)
  truncate!(P;maxm=maxm,cutoff=cutoff,absoluteCutoff=absoluteCutoff,doRelCutoff=doRelCutoff)
  dS = length(P)

  if dS < length(MS)
    MU = MU[:,1:dS]
    resize!(MS,dS)
    MV = MV[:,1:dS]
  end

  u = Index(dS,utags)
  v = u(vtags)
  Uis,Ustore = IndexSet(Lis...,u),Dense{T}(vec(MU))
  #TODO: make a diag storage
  Sis,Sstore = IndexSet(u,v),Dense{Float64}(vec(Matrix(Diagonal(MS))))
  Vis,Vstore = IndexSet(Ris...,v),Dense{T}(Vector{T}(vec(MV)))

  return (Uis,Ustore,Sis,Sstore,Vis,Vstore)
end

function storage_eigen(Astore::T,Lis::IndexSet,Ris::IndexSet,matrixtype::Type{S},truncate::Int,tags::String) where {T<:Dense,S}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MD,MU = eigen(S(reshape(data(Astore),dim_left,dim_right)))

  #TODO: include truncation parameters as keyword arguments
  dim_middle = min(dim_left,dim_right,truncate)
  u = Index(dim_middle,tags)
  v = prime(u)
  Uis,Ustore = IndexSet(Lis...,u),T(vec(MU[:,1:dim_middle]))
  #TODO: make a diag storage
  Dis,Dstore = IndexSet(u,v),T(vec(Matrix(Diagonal(MD[1:dim_middle]))))
  return (Uis,Ustore,Dis,Dstore)
end

function polar(A::Matrix)
  U,S,V = svd(A)
  return U*V',V*Diagonal(S)*V'
end

#TODO: make one generic function storage_factorization(Astore,Lis,Ris,factorization)
function storage_qr(Astore::T,Lis::IndexSet,Ris::IndexSet) where {T<:Dense}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MQ,MP = qr(reshape(data(Astore),dim_left,dim_right))
  dim_middle = min(dim_left,dim_right)
  u = Index(dim_middle,"Link,u")
  #Must call Matrix() on MQ since the QR decomposition outputs a sparse
  #form of the decomposition
  Qis,Qstore = IndexSet(Lis...,u),T(vec(Matrix(MQ)))
  Pis,Pstore = IndexSet(u,Ris...),T(vec(Matrix(MP)))
  return (Qis,Qstore,Pis,Pstore)
end

function storage_polar(Astore::T,Lis::IndexSet,Ris::IndexSet) where {T<:Dense}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MQ,MP = polar(reshape(data(Astore),dim_left,dim_right))
  dim_middle = min(dim_left,dim_right)
  u = Index(dim_middle,"Link,u")
  Qis,Qstore = IndexSet(Lis...,u),T(vec(MQ))
  Pis,Pstore = IndexSet(u,Ris...),T(vec(MP))
  return (Qis,Qstore,Pis,Pstore)
end

