function contract_scalar!(Cdata::CuArray{T},Clabels::Vector{Int},
                          Bdata::CuArray{T},Blabels::Vector{Int},α::Tα,β::Tβ) where {T,Tα<:Number,Tβ<:Number}
  p = calculate_permutation(Blabels,Clabels)
  if β==0
    if is_trivial_permutation(p)
      Cdata .= α.*Bdata
    else
      #TODO: make an optimized permutedims!() that also scales the data
      permutedims!(Cdata,α*Bdata)
    end
  else
    if is_trivial_permutation(p)
      Cdata .= α.*Bdata .+ β.*Cdata
    else
      #TODO: make an optimized permutedims!() that also adds and scales the data
      permBdata = permutedims(Bdata,p)
      Cdata .= α.*permBdata .+ β.*Cdata
    end
  end
  return
end

function contract!(C::CuArray{T},
                   p::CProps,
                   A::CuArray{T},
                   B::CuArray{T},
                   α::Tα=1.0,
                   β::Tβ=0.0) where {T,Tα<:Number,Tβ<:Number}

  # TODO: This is because the permutation convention in C++ ITensor and
  # permutedims in Julia is different
  p.PA = inv(Permutation(p.PA)).data
  p.PB = inv(Permutation(p.PB)).data
  p.PC = inv(Permutation(p.PC)).data
  tA = 'N'
  if p.permuteA
    aref = reshape(permutedims(A,p.PA),p.dmid,p.dleft)
    tA = 'T'
  else
    #A doesn't have to be permuted
    if Atrans(p)
      aref = reshape(A,p.dmid,p.dleft)
      tA = 'T'
    else
      aref = reshape(A,p.dleft,p.dmid)
    end
  end

  tB = 'N'
  if p.permuteB
    bref = reshape(permutedims(B,p.PB),p.dmid,p.dright)
  else
    if Btrans(p)
      bref = reshape(B,p.dright,p.dmid)
      tB = 'T'
    else
      bref = reshape(B,p.dmid,p.dright)
    end
  end

  if p.permuteC
    cref = reshape(C,p.dleft,p.dright)
  else
    if Ctrans(p)
      cref = reshape(C,p.dleft,p.dright)
      if tA=='N' && tB=='N'
        (aref,bref) = (bref,aref)
        tA = tB = 'T'
      elseif tA=='T' && tB=='T'
        (aref,bref) = (bref,aref)
        tA = tB = 'N'
      end
    else
      cref = reshape(C,p.dleft,p.dright)
    end
  end

  CUBLAS.gemm_wrapper!(cref, tA,tB,aref,bref,promote_type(T,Tα)(α),promote_type(T,Tβ)(β))

  if p.permuteC
    permutedims!(C,reshape(cref,p.newCrange...),p.PC)
  end
  return
end

function contract(Cinds::IndexSet,
                  Clabels::Vector{Int},
                  Astore::Dense{SA, TA},
                  Ainds::IndexSet,
                  Alabels::Vector{Int},
                  Bstore::Dense{SB, TB},
                  Binds::IndexSet,
                  Blabels::Vector{Int}) where {SA<:Number,SB<:Number, TA <: CuArray, TB <: CuArray}
  Adims = dims(Ainds)
  Bdims = dims(Binds)
  Cdims = dims(Cinds)

  # Create storage for output tensor
  Cstore = Dense{promote_type(SA,SB), CuVector{promote_type(SA,SB)}}(prod(Cdims))

  Adata = reshape(data(Astore),Adims)
  Bdata = reshape(data(Bstore),Bdims)
  Cdata = reshape(data(Cstore),Cdims)
  contracted = commoninds(Ainds, Binds)
  A_only = uniqueinds(Ainds, Binds)
  B_only = uniqueinds(Binds, Ainds)
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(contracted)
      push!(ind_dict, i)
  end
  if length(A_only) > 0
      for (idx, i) in enumerate(A_only)
          push!(ind_dict, i) 
      end
  end
  if length(B_only) > 0
      for (idx, i) in enumerate(B_only)
          push!(ind_dict, i) 
      end
  end
  ctainds = zeros(Int, length(Ainds))
  ctbinds = zeros(Int, length(Binds))
  ctcinds = zeros(Int, length(Cinds))
  for (ii, ia) in enumerate(Ainds)
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Binds)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  for (ii, ic) in enumerate(Cinds)
      ctcinds[ii] = findfirst(x->x==ic, ind_dict)
  end
  id_op = CuTensor.CUTENSOR_OP_IDENTITY
  CuTensor.contraction!(one(SA), Adata, Vector{Char}(ctainds), id_op, Bdata, Vector{Char}(ctbinds), id_op, one(SB), Cdata, Vector{Char}(ctcinds), id_op, id_op)
  return Cstore
end

