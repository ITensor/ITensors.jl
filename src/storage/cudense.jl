Dense{T, SA}(x::Dense{T, SB}) where {T<:Number, SA<:CuArray, SB<:Array} = Dense{T, S}(CuArray(x))
Dense{T, SA}(x::Dense{T, SB}) where {T<:Number, SA<:Array, SB<:CuArray} = Dense{T, S}(collect(x.data))
Base.collect(x::Dense{T, S}) where {T<:Number, S<:CuArray} = Dense{T, Vector{T}}(collect(x.data))

*(D::Dense{T, AT},x::S) where {T,AT<:CuArray,S<:Number} = Dense{promote_type(T,S), CuVector{promote_type(T,S)}}(x .* data(D))

function storage_contract(Astore::Dense{T, S},
                          Ais::IndexSet,
                          Bstore::Dense{T, S},
                          Bis::IndexSet) where {T, S<:CuArray}
  
  if length(Ais)==0
    Cis = Bis
    Cs = similar(data(Bstore))
    Cstore = Dense{T, S}(mul!(Cs, data(Bstore), data(Astore)))
  elseif length(Bis)==0
    Cis = Ais
    Cs = similar(data(Astore))
    Cstore = Dense{T, S}(mul!(Cs, data(Astore), data(Bstore)))
  else
    #TODO: check for special case when Ais and Bis are disjoint sets
    #I think we should do this analysis outside of storage_contract, at the ITensor level
    #(since it is universal for any storage type and just analyzes in indices)
    (Alabels,Blabels) = compute_contraction_labels(Ais,Bis)
    if is_outer(Alabels,Blabels)
      Cis = IndexSet(Ais,Bis)
      Cstore = outer(Astore,Bstore)
    else
      (Cis,Clabels) = contract_inds(Ais,Alabels,Bis,Blabels)
      Cstore = contract(Cis,Clabels,Astore,Ais,Alabels,Bstore,Bis,Blabels)
    end
  end
  return (Cis,Cstore)
end

function storage_svd(Astore::Dense{T, S},
                     Lis::IndexSet,
                     Ris::IndexSet;
                     kwargs...
                    ) where {T, S<:CuArray}
  maxm::Int = get(kwargs,:maxm,min(dim(Lis),dim(Ris)))
  minm::Int = get(kwargs,:minm,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
  utags::String = get(kwargs,:utags,"Link,u")
  vtags::String = get(kwargs,:vtags,"Link,v")
  rsd = reshape(data(Astore),dim(Lis),dim(Ris))
  MU,MS,MV = CUSOLVER.svd(rsd)

  sqr(x) = x^2
  P = collect(sqr.(MS))
  truncate!(P;maxm=maxm,cutoff=cutoff,absoluteCutoff=absoluteCutoff,doRelCutoff=doRelCutoff)
  dS = length(P)
  if dS < length(MS)
    MU = MU[:,1:dS]
    MS = MS[1:dS]
    MV = MV[:,1:dS]
  end

  u = Index(dS,utags)
  v = settags(u,vtags)
  Uis,Ustore = IndexSet(Lis...,u),Dense{T, CuVector{T}}(vec(MU))
  #TODO: make a diag storage
  Sis,Sstore = IndexSet(u,v),Dense{T, CuVector{T}}(vec(CuMatrix(Diagonal(MS))))
  Vis,Vstore = IndexSet(Ris...,v),Dense{T, CuVector{T}}(CuVector{T}(vec(MV)))

  return (Uis,Ustore,Sis,Sstore,Vis,Vstore)
end

function storage_eigen(Astore::Dense{S, T}, Lis::IndexSet,Ris::IndexSet,matrixtype::Type{T},truncate::Int,tags::String) where {T<:CuArray,S<:Number}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  local d_W, d_V
  d_A = reshape(data(Astore),dim_left,dim_right)
  if( S <: Complex )
    d_W, d_V   = CUSOLVER.heevd!('V','U', d_A)
  else
    d_W, d_V   = CUSOLVER.syevd!('V','U', d_A)
  end
  #TODO: include truncation parameters as keyword arguments
  dim_middle = min(dim_left,dim_right,truncate)
  u = Index(dim_middle,tags)
  v = prime(u)
  Uis,Ustore = IndexSet(Lis...,u),Dense{S, T}(vec(d_V[:,1:dim_middle]))
  #TODO: make a diag storage
  Dis,Dstore = IndexSet(u,v),Dense{S, T}(vec(Matrix(Diagonal(d_W[1:dim_middle]))))
  return (Uis,Ustore,Dis,Dstore)
end

function storage_qr(Astore::Dense{S, T},Lis::IndexSet,Ris::IndexSet) where {T<:CuArray, S<:Number}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  dQR = qr!(reshape(data(Astore),dim_left,dim_right))
  MQ = dQR.Q
  MP = dQR.R
  dim_middle = min(dim_left,dim_right)
  u = Index(dim_middle,"Link,u")
  #Must call Matrix() on MQ since the QR decomposition outputs a sparse
  #form of the decomposition
  Qis,Qstore = IndexSet(Lis...,u),Dense{S, T}(vec(CuArray(MQ)))
  Pis,Pstore = IndexSet(u,Ris...),Dense{S, T}(vec(CuArray(MP)))
  return (Qis,Qstore,Pis,Pstore)
end

function storage_polar(Astore::Dense{S, T},Lis::IndexSet,Ris::IndexSet) where {T<:CuArray, S<:Number}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MQ,MP = polar(reshape(data(Astore),dim_left,dim_right))
  dim_middle = min(dim_left,dim_right)
  #u = Index(dim_middle,"Link,u")
  Uis = addtags(Ris,"u")
  Qis,Qstore = IndexSet(Lis...,Uis...),Dense{S, T}(vec(MQ))
  Pis,Pstore = IndexSet(Uis...,Ris...),Dense{S, T}(vec(MP))
  return (Qis,Qstore,Pis,Pstore)
end

function storage_add!(Bstore::Dense{SB, T},Bis::IndexSet,Astore::Dense{SA, T},Ais::IndexSet) where {T<:CuArray, SA<:Number, SB<:Number}
  Adata = reshape(data(Astore), dims(Ais))
  p = ITensors.calculate_permutation(Bis,Ais)
  #permAdata = permutedims(reshape(Adata,dims(Ais)),p)
  Bdata = reshape(data(Bstore), dims(Bis))
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Bis)
      push!(ind_dict, i)
  end
  id_op = CuTensor.CUTENSOR_OP_IDENTITY
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  ctcinds = copy(ctbinds)
  C = cuzeros(SB, size(Bdata))
  CuTensor.elementwiseBinary!(one(SA), Adata, Vector{Char}(ctainds), id_op, one(SB), Bdata, Vector{Char}(ctbinds), id_op, C, Vector{Char}(ctcinds), CuTensor.CUTENSOR_OP_ADD)
  copyto!(Bstore.data, reshape(C, length(Bdata)))
  return Bstore
end

function storage_permute!(Bstore::Dense{SB, T},Bis::IndexSet,Astore::Dense{SB, T},Ais::IndexSet) where {T<:CuArray, SA<:Number, SB<:Number}
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Ais)
      push!(ind_dict, i)
  end
  Adata = data(Astore)
  Bdata = data(Bstore)
  reshapeBdata = reshape(Bdata,dims(Bis))
  reshapeAdata = reshape(Adata,dims(Ais))
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  CuTensor.permutation!(one(eltype(Adata)), reshapeAdata, Vector{Char}(ctainds), reshapeBdata, Vector{Char}(ctbinds)) 
  copyto!(Bstore.data, reshape(reshapeBdata, length(Bstore.data)))
end
