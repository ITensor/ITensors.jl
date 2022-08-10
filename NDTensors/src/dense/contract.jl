function contraction_output(
  T1::DenseTensor, T2::DenseTensor, indsR
)
  type_R = contraction_output_type(typeof(T1), typeof(T2), typeof(indsR))
  return similar(type_R, indsR)
end

Strided.StridedView(T::DenseTensor) = StridedView(convert(Array, T))

function contract!(
  C::DenseTensor,
  labelsC,
  A::DenseTensor,
  labelsA,
  B::DenseTensor,
  labelsB,
  α::Number=true,
  β::Number=false,
) where {ElT<:Number}
  props = ContractionProperties(labelsA, labelsB, labelsC)
  compute_contraction_properties!(props, size(A), size(B), size(C))
  contract!(props, C, A, B, α, β)
  return C
end

function contract!(
  props::ContractionProperties,
  C::DenseTensor,
  A::DenseTensor,
  B::DenseTensor,
  α::Number=true,
  β::Number=false,
) where {ElT<:Number}
  contract!(props, array(C), array(A), array(B), α, β)
  return C
end

function contract!(
  props::ContractionProperties,
  C::AbstractArray,
  A::AbstractArray,
  B::AbstractArray,
  α::Number=true,
  β::Number=false,
)
  T = promote_type(eltype(A), eltype(B))
  A2 = convert(AbstractArray{T}, A)
  B2 = convert(AbstractArray{T}, B)
  contract!(props, C, A2, B2, α, β)
end

function contract!(
  props::ContractionProperties,
  C::AbstractArray{ElT},
  A::AbstractArray{ElT},
  B::AbstractArray{ElT},
  α::Number=true,
  β::Number=false,
) where {ElT<:Number}
  AM, tA, BM, tB, CM = reshape_to_matmul(props, C, A, B)
  BLAS.gemm!(tA, tB, α, AM, BM, β, CM)
  if props.permuteC
    # pC = NTuple{ndims(C),Int}(props.PC)
    # Cr = reshape(CM, props.newCrange)
    permutedims!(C, reshape(CM, props.newCrange), props.PC)
  end
  return C
end

function reshape_to_matmul(
  props::ContractionProperties,
  C::AbstractArray,
  A::AbstractArray,
  B::AbstractArray,
)
  tA = 'N'
  if props.permuteA
    # pA = NTuple{ndims(A),Int}(props.PA)
    AM = reshape(permutedims(A, props.PA), (props.dmid, props.dleft))
    tA = 'T'
  else
    #A doesn't have to be permuted
    if Atrans(props)
      AM = reshape(A, (props.dmid, props.dleft))
      tA = 'T'
    else
      AM = reshape(A, (props.dleft, props.dmid))
    end
  end

  tB = 'N'
  if props.permuteB
    # pB = NTuple{ndims(B),Int}(props.PB)
    BM = reshape(permutedims(B, props.PB), (props.dmid, props.dright))
  else
    if Btrans(props)
      BM = reshape(B, (props.dright, props.dmid))
      tB = 'T'
    else
      BM = reshape(B, (props.dmid, props.dright))
    end
  end

  # TODO: this logic may be wrong
  if props.permuteC
    # Need to copy here since we will be permuting
    # into C later
    CM = reshape(copy(C), (props.dleft, props.dright))
  else
    if Ctrans(props)
      CM = reshape(C, (props.dright, props.dleft))
      (AM, BM) = (BM, AM)
      if tA == tB
        tA = tB = (tA == 'T' ? 'N' : 'T')
      end
    else
      CM = reshape(C, (props.dleft, props.dright))
    end
  end
  return AM, tA, BM, tB, CM
end

## function outer!(
##   R::DenseTensor{ElR}, T1::DenseTensor{ElT1}, T2::DenseTensor{ElT2}
## ) where {ElR,ElT1,ElT2}
##   if ElT1 != ElT2
##     # TODO: use promote instead
##     # T1,T2 = promote(T1,T2)
## 
##     ElT1T2 = promote_type(ElT1, ElT2)
##     if ElT1 != ElT1T2
##       # TODO: get this working
##       # T1 = ElR.(T1)
##       T1 = one(ElT1T2) * T1
##     end
##     if ElT2 != ElT1T2
##       # TODO: get this working
##       # T2 = ElR.(T2)
##       T2 = one(ElT1T2) * T2
##     end
##   end
## 
##   v1 = data(T1)
##   v2 = data(T2)
##   RM = reshape(R, length(v1), length(v2))
##   #RM .= v1 .* transpose(v2)
##   #mul!(RM, v1, transpose(v2))
##   _gemm!('N', 'T', one(ElR), v1, v2, zero(ElR), RM)
##   return R
## end
## 
## export backend_auto, backend_blas, backend_generic
## 
## @eval struct GemmBackend{T}
##   (f::Type{<:GemmBackend})() = $(Expr(:new, :f))
## end
## GemmBackend(s) = GemmBackend{Symbol(s)}()
## macro GemmBackend_str(s)
##   return :(GemmBackend{$(Expr(:quote, Symbol(s)))})
## end
## 
## const gemm_backend = Ref(:Auto)
## function backend_auto()
##   return gemm_backend[] = :Auto
## end
## function backend_blas()
##   return gemm_backend[] = :BLAS
## end
## function backend_generic()
##   return gemm_backend[] = :Generic
## end
## 
## @inline function auto_select_backend(
##   ::Type{<:StridedVecOrMat{<:BlasFloat}},
##   ::Type{<:StridedVecOrMat{<:BlasFloat}},
##   ::Type{<:StridedVecOrMat{<:BlasFloat}},
## )
##   return GemmBackend(:BLAS)
## end
## 
## @inline function auto_select_backend(
##   ::Type{<:AbstractVecOrMat}, ::Type{<:AbstractVecOrMat}, ::Type{<:AbstractVecOrMat}
## )
##   return GemmBackend(:Generic)
## end
## 
## function _gemm!(
##   tA, tB, alpha, A::TA, B::TB, beta, C::TC
## ) where {TA<:AbstractVecOrMat,TB<:AbstractVecOrMat,TC<:AbstractVecOrMat}
##   if gemm_backend[] == :Auto
##     _gemm!(auto_select_backend(TA, TB, TC), tA, tB, alpha, A, B, beta, C)
##   else
##     _gemm!(GemmBackend(gemm_backend[]), tA, tB, alpha, A, B, beta, C)
##   end
## end
## 
## # BLAS matmul
## function _gemm!(
##   ::GemmBackend{:BLAS},
##   tA,
##   tB,
##   alpha,
##   A::AbstractVecOrMat,
##   B::AbstractVecOrMat,
##   beta,
##   C::AbstractVecOrMat,
## )
##   #@timeit_debug timer "BLAS.gemm!" begin
##   return BLAS.gemm!(tA, tB, alpha, A, B, beta, C)
##   #end # @timeit
## end
## 
## # generic matmul
## function _gemm!(
##   ::GemmBackend{:Generic},
##   tA,
##   tB,
##   alpha::AT,
##   A::AbstractVecOrMat,
##   B::AbstractVecOrMat,
##   beta::BT,
##   C::AbstractVecOrMat,
## ) where {AT,BT}
##   mul!(C, tA == 'T' ? transpose(A) : A, tB == 'T' ? transpose(B) : B, alpha, beta)
##   return C
## end
## 
## # TODO: call outer!!, make this generic
## function outer(T1::DenseTensor{ElT1}, T2::DenseTensor{ElT2}) where {ElT1,ElT2}
##   array_outer = vec(array(T1)) * transpose(vec(array(T2)))
##   inds_outer = unioninds(inds(T1), inds(T2))
##   return tensor(Dense{promote_type(ElT1, ElT2)}(vec(array_outer)), inds_outer)
## end

## # Both are scalar-like tensors
## function _contract_scalar!(
##   R::DenseTensor{ElR},
##   labelsR,
##   T1::Number,
##   labelsT1,
##   T2::Number,
##   labelsT2,
##   α=one(ElR),
##   β=zero(ElR),
## ) where {ElR}
##   if iszero(β)
##     R[1] = α * T1 * T2
##   elseif iszero(α)
##     R[1] = β * R[1]
##   else
##     R[1] = α * T1 * T2 + β * R[1]
##   end
##   return R
## end
## 
## # Trivial permutation
## # Version where R and T have different element types, so we can't call BLAS
## # Instead use Julia's broadcasting (maybe consider Strided in the future)
## function _contract_scalar_noperm!(
##   R::DenseTensor{ElR}, T::DenseTensor, α, β=zero(ElR)
## ) where {ElR}
##   Rᵈ = data(R)
##   Tᵈ = data(T)
##   if iszero(β)
##     if iszero(α)
##       fill!(Rᵈ, 0)
##     else
##       Rᵈ .= α .* Tᵈ
##     end
##   elseif isone(β)
##     if iszero(α)
##       # No-op
##       # Rᵈ .= Rᵈ
##     else
##       Rᵈ .= α .* Tᵈ .+ Rᵈ
##     end
##   else
##     if iszero(α)
##       # Rᵈ .= β .* Rᵈ
##       BLAS.scal!(length(Rᵈ), β, Rᵈ, 1)
##     else
##       Rᵈ .= α .* Tᵈ .+ β .* Rᵈ
##     end
##   end
##   return R
## end
## 
## # Trivial permutation
## # Version where R and T are the same element type, so we can
## # call BLAS
## function _contract_scalar_noperm!(
##   R::DenseTensor{ElR}, T::DenseTensor{ElR}, α, β=zero(ElR)
## ) where {ElR}
##   Rᵈ = data(R)
##   Tᵈ = data(T)
##   if iszero(β)
##     if iszero(α)
##       fill!(Rᵈ, 0)
##     else
##       # Rᵈ .= α .* T₂ᵈ
##       BLAS.axpby!(α, Tᵈ, β, Rᵈ)
##     end
##   elseif isone(β)
##     if iszero(α)
##       # No-op
##       # Rᵈ .= Rᵈ
##     else
##       # Rᵈ .= α .* Tᵈ .+ Rᵈ
##       BLAS.axpy!(α, Tᵈ, Rᵈ)
##     end
##   else
##     if iszero(α)
##       # Rᵈ .= β .* Rᵈ
##       BLAS.scal!(length(Rᵈ), β, Rᵈ, 1)
##     else
##       # Rᵈ .= α .* Tᵈ .+ β .* Rᵈ
##       BLAS.axpby!(α, Tᵈ, β, Rᵈ)
##     end
##   end
##   return R
## end
## 
## # Non-trivial permutation
## function _contract_scalar_perm!(
##   Rᵃ::AbstractArray{ElR}, Tᵃ::AbstractArray, perm, α, β=zero(ElR)
## ) where {ElR}
##   if iszero(β)
##     if iszero(α)
##       fill!(Rᵃ, 0)
##     else
##       @strided Rᵃ .= α .* permutedims(Tᵃ, perm)
##     end
##   elseif isone(β)
##     if iszero(α)
##       # Rᵃ .= Rᵃ
##       # No-op
##     else
##       @strided Rᵃ .= α .* permutedims(Tᵃ, perm) .+ Rᵃ
##     end
##   else
##     if iszero(α)
##       # Rᵃ .= β .* Rᵃ
##       BLAS.scal!(length(Rᵃ), β, Rᵃ, 1)
##     else
##       Rᵃ .= α .* permutedims(Tᵃ, perm) .+ β .* Rᵃ
##     end
##   end
##   return Rᵃ
## end
## 
## function drop_singletons(::Order{N}, labels, dims) where {N}
##   labelsᵣ = ntuple(zero, Val(N))
##   dimsᵣ = labelsᵣ
##   nkeep = 1
##   for n in 1:length(dims)
##     if dims[n] > 1
##       labelsᵣ = @inbounds setindex(labelsᵣ, labels[n], nkeep)
##       dimsᵣ = @inbounds setindex(dimsᵣ, dims[n], nkeep)
##       nkeep += 1
##     end
##   end
##   return labelsᵣ, dimsᵣ
## end
## 
## function _contract_scalar_maybe_perm!(
##   ::Order{N}, R::DenseTensor{ElR,NR}, labelsR, T::DenseTensor, labelsT, α, β=zero(ElR)
## ) where {ElR,NR,N}
##   labelsRᵣ, dimsRᵣ = drop_singletons(Order(N), labelsR, dims(R))
##   labelsTᵣ, dimsTᵣ = drop_singletons(Order(N), labelsT, dims(T))
##   perm = getperm(labelsRᵣ, labelsTᵣ)
##   if is_trivial_permutation(perm)
##     # trivial permutation
##     _contract_scalar_noperm!(R, T, α, β)
##   else
##     # non-trivial permutation
##     Rᵣ = ReshapedArray(data(R), dimsRᵣ, ())
##     Tᵣ = ReshapedArray(data(T), dimsTᵣ, ())
##     _contract_scalar_perm!(Rᵣ, Tᵣ, perm, α, β)
##   end
##   return R
## end
## 
## function _contract_scalar_maybe_perm!(
##   R::DenseTensor{ElR,NR}, labelsR, T::DenseTensor, labelsT, α, β=zero(ElR)
## ) where {ElR,NR}
##   N = count(≠(1), dims(R))
##   _contract_scalar_maybe_perm!(Order(N), R, labelsR, T, labelsT, α, β)
##   return R
## end
## 
## # XXX: handle case of non-trivial permutation
## function _contract_scalar_maybe_perm!(
##   R::DenseTensor{ElR,NR},
##   labelsR,
##   T₁::DenseTensor,
##   labelsT₁,
##   T₂::DenseTensor,
##   labelsT₂,
##   α=one(ElR),
##   β=zero(ElR),
## ) where {ElR,NR}
##   if nnz(T₁) == 1
##     _contract_scalar_maybe_perm!(R, labelsR, T₂, labelsT₂, α * T₁[1], β)
##   elseif nnz(T₂) == 1
##     _contract_scalar_maybe_perm!(R, labelsR, T₁, labelsT₁, α * T₂[1], β)
##   else
##     error("In _contract_scalar_perm!, one tensor must be a scalar")
##   end
##   return R
## end
## 
## # At least one of the tensors is size 1
## function _contract_scalar!(
##   R::DenseTensor{ElR},
##   labelsR,
##   T1::DenseTensor,
##   labelsT1,
##   T2::DenseTensor,
##   labelsT2,
##   α=one(ElR),
##   β=zero(ElR),
## ) where {ElR}
##   if nnz(T1) == nnz(T2) == 1
##     _contract_scalar!(R, labelsR, T1[1], labelsT1, T2[1], labelsT2, α, β)
##   else
##     _contract_scalar_maybe_perm!(R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
##   end
##   return R
## end

## function contract!(
##   R::DenseTensor{ElR,NR},
##   labelsR,
##   T1::DenseTensor{ElT1,N1},
##   labelsT1,
##   T2::DenseTensor{ElT2,N2},
##   labelsT2
## ) where {ElR,ElT1,ElT2,NR,N1,N2}
##   # Special case for scalar tensors
##   if nnz(T1) == 1 || nnz(T2) == 1
##     _contract_scalar!(R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
##     return R
##   end
## 
##   if using_tblis() && ElR <: LinearAlgebra.BlasReal && (ElR == ElT1 == ElT2 == Elα == Elβ)
##     #@timeit_debug timer "TBLIS contract!" begin
##     contract!(Val(:TBLIS), R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
##     #end
##     return R
##   end
## 
##   if N1 + N2 == NR
##     outer!(R, T1, T2)
##     labelsRp = (labelsT1..., labelsT2...)
##     perm = getperm(labelsR, labelsRp)
##     if !is_trivial_permutation(perm)
##       permutedims!(R, copy(R), perm)
##     end
##     return R
##   end
## 
##   props = ContractionProperties(labelsT1, labelsT2, labelsR)
##   compute_contraction_properties!(props, T1, T2, R)
## 
##   if ElT1 != ElT2
##     # TODO: use promote instead
##     # T1, T2 = promote(T1, T2)
## 
##     ElT1T2 = promote_type(ElT1, ElT2)
##     if ElT1 != ElR
##       # TODO: get this working
##       # T1 = ElR.(T1)
##       T1 = one(ElT1T2) * T1
##     end
##     if ElT2 != ElR
##       # TODO: get this working
##       # T2 = ElR.(T2)
##       T2 = one(ElT1T2) * T2
##     end
##   end
## 
##   _contract!(R, T1, T2, props, α, β)
##   return R
## end

