
Strided.StridedView(T::DenseTensor) = StridedView(convert(Array, T))

function drop_singletons(::Order{N}, labels, dims) where {N}
  labelsᵣ = ntuple(zero, Val(N))
  dimsᵣ = labelsᵣ
  nkeep = 1
  for n in 1:length(dims)
    if dims[n] > 1
      labelsᵣ = @inbounds setindex(labelsᵣ, labels[n], nkeep)
      dimsᵣ = @inbounds setindex(dimsᵣ, dims[n], nkeep)
      nkeep += 1
    end
  end
  return labelsᵣ, dimsᵣ
end

# svd of an order-n tensor according to positions Lpos
# and Rpos
function LinearAlgebra.svd(
  T::DenseTensor{<:Number,N,IndsT}, Lpos::NTuple{NL,Int}, Rpos::NTuple{NR,Int}; kwargs...
) where {N,IndsT,NL,NR}
  M = permute_reshape(T, Lpos, Rpos)
  UM, S, VM, spec = svd(M; kwargs...)
  u = ind(UM, 2)
  v = ind(VM, 2)

  Linds = similartype(IndsT, Val{NL})(ntuple(i -> inds(T)[Lpos[i]], Val(NL)))
  Uinds = push(Linds, u)

  # TODO: do these positions need to be reversed?
  Rinds = similartype(IndsT, Val{NR})(ntuple(i -> inds(T)[Rpos[i]], Val(NR)))
  Vinds = push(Rinds, v)

  U = reshape(UM, Uinds)
  V = reshape(VM, Vinds)

  return U, S, V, spec
end

# qr decomposition of an order-n tensor according to 
# positions Lpos and Rpos
function LinearAlgebra.qr(
  T::DenseTensor{<:Number,N,IndsT}, Lpos::NTuple{NL,Int}, Rpos::NTuple{NR,Int}; kwargs...
) where {N,IndsT,NL,NR}
  M = permute_reshape(T, Lpos, Rpos)
  QM, RM = qr(M; kwargs...)
  q = ind(QM, 2)
  r = ind(RM, 1)
  # TODO: simplify this by permuting inds(T) by (Lpos,Rpos)
  # then grab Linds,Rinds
  Linds = similartype(IndsT, Val{NL})(ntuple(i -> inds(T)[Lpos[i]], Val(NL)))
  Qinds = push(Linds, r)
  Q = reshape(QM, Qinds)
  Rinds = similartype(IndsT, Val{NR})(ntuple(i -> inds(T)[Rpos[i]], Val(NR)))
  Rinds = pushfirst(Rinds, r)
  R = reshape(RM, Rinds)
  return Q, R
end

# polar decomposition of an order-n tensor according to positions Lpos
# and Rpos
function polar(
  T::DenseTensor{<:Number,N,IndsT}, Lpos::NTuple{NL,Int}, Rpos::NTuple{NR,Int}
) where {N,IndsT,NL,NR}
  M = permute_reshape(T, Lpos, Rpos)
  UM, PM = polar(M)

  # TODO: turn these into functions
  Linds = similartype(IndsT, Val{NL})(ntuple(i -> inds(T)[Lpos[i]], Val(NL)))
  Rinds = similartype(IndsT, Val{NR})(ntuple(i -> inds(T)[Rpos[i]], Val(NR)))

  # Use sim to create "similar" indices, in case
  # the indices have identifiers. If not this should
  # act as an identity operator
  simRinds = sim(Rinds)
  Uinds = (Linds..., simRinds...)
  Pinds = (simRinds..., Rinds...)

  U = reshape(UM, Uinds)
  P = reshape(PM, Pinds)
  return U, P
end

function LinearAlgebra.exp(
  T::DenseTensor{ElT,N}, Lpos::NTuple{NL,Int}, Rpos::NTuple{NR,Int}; ishermitian::Bool=false
) where {ElT,N,NL,NR}
  M = permute_reshape(T, Lpos, Rpos)
  indsTp = permute(inds(T), (Lpos..., Rpos...))
  if ishermitian
    expM = parent(exp(Hermitian(matrix(M))))
    return tensor(Dense{ElT}(vec(expM)), indsTp)
  else
    expM = exp(M)
    return reshape(expM, indsTp)
  end
end
