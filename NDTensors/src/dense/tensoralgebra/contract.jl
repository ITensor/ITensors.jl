export backend_auto, backend_blas, backend_generic

@eval struct GemmBackend{T}
  (f::Type{<:GemmBackend})() = $(Expr(:new, :f))
end
GemmBackend(s) = GemmBackend{Symbol(s)}()
macro GemmBackend_str(s)
  return :(GemmBackend{$(Expr(:quote, Symbol(s)))})
end

const gemm_backend = Ref(:Auto)
function backend_auto()
  return gemm_backend[] = :Auto
end
function backend_blas()
  return gemm_backend[] = :BLAS
end
function backend_generic()
  return gemm_backend[] = :Generic
end

@inline function auto_select_backend(
  ::Type{<:StridedVecOrMat{<:BlasFloat}},
  ::Type{<:StridedVecOrMat{<:BlasFloat}},
  ::Type{<:StridedVecOrMat{<:BlasFloat}},
)
  return GemmBackend(:BLAS)
end

@inline function auto_select_backend(
  ::Type{<:AbstractVecOrMat}, ::Type{<:AbstractVecOrMat}, ::Type{<:AbstractVecOrMat}
)
  return GemmBackend(:Generic)
end

function _gemm!(
  tA, tB, alpha, A::TA, B::TB, beta, C::TC
) where {TA<:AbstractVecOrMat,TB<:AbstractVecOrMat,TC<:AbstractVecOrMat}
  if gemm_backend[] == :Auto
    _gemm!(auto_select_backend(TA, TB, TC), tA, tB, alpha, A, B, beta, C)
  else
    _gemm!(GemmBackend(gemm_backend[]), tA, tB, alpha, A, B, beta, C)
  end
end

# BLAS matmul
function _gemm!(
  ::GemmBackend{:BLAS},
  tA,
  tB,
  alpha,
  A::AbstractVecOrMat,
  B::AbstractVecOrMat,
  beta,
  C::AbstractVecOrMat,
)
  #@timeit_debug timer "BLAS.gemm!" begin
  return BLAS.gemm!(tA, tB, alpha, A, B, beta, C)
  #end # @timeit
end

# generic matmul
function _gemm!(
  ::GemmBackend{:Generic},
  tA,
  tB,
  alpha::AT,
  A::AbstractVecOrMat,
  B::AbstractVecOrMat,
  beta::BT,
  C::AbstractVecOrMat,
) where {AT,BT}
  mul!(C, tA == 'T' ? transpose(A) : A, tB == 'T' ? transpose(B) : B, alpha, beta)
  return C
end

function contraction_output(tensor1::DenseTensor, tensor2::DenseTensor, indsR)
  tensortypeR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  return NDTensors.similar(tensortypeR, indsR)
end

# Both are scalar-like tensors
function _contract_scalar!(
  R::DenseTensor{ElR},
  labelsR,
  T1::Number,
  labelsT1,
  T2::Number,
  labelsT2,
  α=one(ElR),
  β=zero(ElR),
) where {ElR}
  if iszero(β)
    R[1] = α * T1 * T2
  elseif iszero(α)
    R[1] = β * R[1]
  else
    R[1] = α * T1 * T2 + β * R[1]
  end
  return R
end

# Trivial permutation
# Version where R and T have different element types, so we can't call BLAS
# Instead use Julia's broadcasting (maybe consider Strided in the future)
function _contract_scalar_noperm!(
  R::DenseTensor{ElR}, T::DenseTensor, α, β=zero(ElR)
) where {ElR}
  Rᵈ = data(R)
  Tᵈ = data(T)
  if iszero(β)
    if iszero(α)
      fill!(Rᵈ, 0)
    else
      Rᵈ .= α .* Tᵈ
    end
  elseif isone(β)
    if iszero(α)
      # No-op
      # Rᵈ .= Rᵈ
    else
      Rᵈ .= α .* Tᵈ .+ Rᵈ
    end
  else
    if iszero(α)
      # Rᵈ .= β .* Rᵈ
      BLAS.scal!(length(Rᵈ), β, Rᵈ, 1)
    else
      Rᵈ .= α .* Tᵈ .+ β .* Rᵈ
    end
  end
  return R
end

# Trivial permutation
# Version where R and T are the same element type, so we can
# call BLAS
function _contract_scalar_noperm!(
  R::DenseTensor{ElR}, T::DenseTensor{ElR}, α, β=zero(ElR)
) where {ElR}
  Rᵈ = data(R)
  Tᵈ = data(T)
  if iszero(β)
    if iszero(α)
      fill!(Rᵈ, 0)
    else
      # Rᵈ .= α .* T₂ᵈ
      LinearAlgebra.axpby!(α, Tᵈ, β, Rᵈ)
    end
  elseif isone(β)
    if iszero(α)
      # No-op
      # Rᵈ .= Rᵈ
    else
      # Rᵈ .= α .* Tᵈ .+ Rᵈ
      LinearAlgebra.axpy!(α, Tᵈ, Rᵈ)
    end
  else
    if iszero(α)
      # Rᵈ .= β .* Rᵈ
      LinearAlgebra.scal!(length(Rᵈ), β, Rᵈ, 1)
    else
      # Rᵈ .= α .* Tᵈ .+ β .* Rᵈ
      LinearAlgebra.axpby!(α, Tᵈ, β, Rᵈ)
    end
  end
  return R
end

# Non-trivial permutation
function _contract_scalar_perm!(
  Rᵃ::AbstractArray{ElR}, Tᵃ::AbstractArray, perm, α, β=zero(ElR)
) where {ElR}
  if iszero(β)
    if iszero(α)
      fill!(Rᵃ, 0)
    else
      @strided Rᵃ .= α .* permutedims(Tᵃ, perm)
    end
  elseif isone(β)
    if iszero(α)
      # Rᵃ .= Rᵃ
      # No-op
    else
      @strided Rᵃ .= α .* permutedims(Tᵃ, perm) .+ Rᵃ
    end
  else
    if iszero(α)
      # Rᵃ .= β .* Rᵃ
      LinearAlgebra.scal!(length(Rᵃ), β, Rᵃ, 1)
    else
      Rᵃ .= α .* permutedims(Tᵃ, perm) .+ β .* Rᵃ
    end
  end
  return Rᵃ
end

function _contract_scalar_maybe_perm!(
  ::Order{N}, R::DenseTensor{ElR,NR}, labelsR, T::DenseTensor, labelsT, α, β=zero(ElR)
) where {ElR,NR,N}
  labelsRᵣ, dimsRᵣ = drop_singletons(Order(N), labelsR, dims(R))
  labelsTᵣ, dimsTᵣ = drop_singletons(Order(N), labelsT, dims(T))
  perm = getperm(labelsRᵣ, labelsTᵣ)
  if is_trivial_permutation(perm)
    # trivial permutation
    _contract_scalar_noperm!(R, T, α, β)
  else
    # non-trivial permutation
    Rᵣ = ReshapedArray(data(R), dimsRᵣ, ())
    Tᵣ = ReshapedArray(data(T), dimsTᵣ, ())
    _contract_scalar_perm!(Rᵣ, Tᵣ, perm, α, β)
  end
  return R
end

function _contract_scalar_maybe_perm!(
  R::DenseTensor{ElR,NR}, labelsR, T::DenseTensor, labelsT, α, β=zero(ElR)
) where {ElR,NR}
  N = count(≠(1), dims(R))
  _contract_scalar_maybe_perm!(Order(N), R, labelsR, T, labelsT, α, β)
  return R
end

# XXX: handle case of non-trivial permutation
function _contract_scalar_maybe_perm!(
  R::DenseTensor{ElR,NR},
  labelsR,
  T₁::DenseTensor,
  labelsT₁,
  T₂::DenseTensor,
  labelsT₂,
  α=one(ElR),
  β=zero(ElR),
) where {ElR,NR}
  if nnz(T₁) == 1
    _contract_scalar_maybe_perm!(R, labelsR, T₂, labelsT₂, α * T₁[1], β)
  elseif nnz(T₂) == 1
    _contract_scalar_maybe_perm!(R, labelsR, T₁, labelsT₁, α * T₂[1], β)
  else
    error("In _contract_scalar_perm!, one tensor must be a scalar")
  end
  return R
end

# At least one of the tensors is size 1
function _contract_scalar!(
  R::DenseTensor{ElR},
  labelsR,
  T1::DenseTensor,
  labelsT1,
  T2::DenseTensor,
  labelsT2,
  α=one(ElR),
  β=zero(ElR),
) where {ElR}
  if nnz(T1) == nnz(T2) == 1
    _contract_scalar!(R, labelsR, T1[1], labelsT1, T2[1], labelsT2, α, β)
  else
    _contract_scalar_maybe_perm!(R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
  end
  return R
end

function contract!(
  R::DenseTensor{ElR,NR},
  labelsR,
  T1::DenseTensor{ElT1,N1},
  labelsT1,
  T2::DenseTensor{ElT2,N2},
  labelsT2,
  α::Elα=one(ElR),
  β::Elβ=zero(ElR),
) where {Elα,Elβ,ElR,ElT1,ElT2,NR,N1,N2}
  # Special case for scalar tensors
  if nnz(T1) == 1 || nnz(T2) == 1
    _contract_scalar!(R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
    return R
  end

  if using_tblis() && ElR <: LinearAlgebra.BlasReal && (ElR == ElT1 == ElT2 == Elα == Elβ)
    #@timeit_debug timer "TBLIS contract!" begin
    contract!(Val(:TBLIS), R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
    #end
    return R
  end

  if N1 + N2 == NR
    outer!(R, T1, T2)
    labelsRp = (labelsT1..., labelsT2...)
    perm = getperm(labelsR, labelsRp)
    if !is_trivial_permutation(perm)
      permutedims!(R, copy(R), perm)
    end
    return R
  end

  props = ContractionProperties(labelsT1, labelsT2, labelsR)
  compute_contraction_properties!(props, T1, T2, R)

  if ElT1 != ElT2
    # TODO: use promote instead
    # T1, T2 = promote(T1, T2)

    ElT1T2 = promote_type(ElT1, ElT2)
    if ElT1 != ElR
      # TODO: get this working
      # T1 = ElR.(T1)
      T1 = one(ElT1T2) * T1
    end
    if ElT2 != ElR
      # TODO: get this working
      # T2 = ElR.(T2)
      T2 = one(ElT1T2) * T2
    end
  end

  _contract!(R, T1, T2, props, α, β)
  return R
  #end
end

function _contract!(
  CT::DenseTensor{El,NC},
  AT::DenseTensor{El,NA},
  BT::DenseTensor{El,NB},
  props::ContractionProperties,
  α::Number=one(El),
  β::Number=zero(El),
) where {El,NC,NA,NB}
  C = array(CT)
  A = array(AT)
  B = array(BT)

  return _contract!(C, A, B, props, α, β)
end

function _contract!(
  CT::AbstractArray{El,NC},
  AT::AbstractArray{El,NA},
  BT::AbstractArray{El,NB},
  props::ContractionProperties,
  α::Number=one(El),
  β::Number=zero(El),
) where {El,NC,NA,NB}
  tA = 'N'
  if props.permuteA
    #@timeit_debug timer "_contract!: permutedims A" begin
    @strided Ap = permutedims(AT, props.PA)
    #end # @timeit
    AM = transpose(reshape(Ap, (props.dmid, props.dleft)))
  else
    #A doesn't have to be permuted
    if Atrans(props)
      AM = transpose(reshape(AT, (props.dmid, props.dleft)))
    else
      AM = reshape(AT, (props.dleft, props.dmid))
    end
  end

  tB = 'N'
  if props.permuteB
    #@timeit_debug timer "_contract!: permutedims B" begin
    @strided Bp = permutedims(BT, props.PB)
    #end # @timeit
    BM = reshape(Bp, (props.dmid, props.dright))
  else
    if Btrans(props)
      BM = transpose(reshape(BT, (props.dright, props.dmid)))
    else
      BM = reshape(BT, (props.dmid, props.dright))
    end
  end

  # TODO: this logic may be wrong
  if props.permuteC
    # if we are computing C = α * A B + β * C
    # we need to make sure C is permuted to the same 
    # ordering as A B which is the inverse of props.PC
    if β ≠ 0
      CM = reshape(permutedims(CT, invperm(props.PC)), (props.dleft, props.dright))
    else
      # Need to copy here since we will be permuting
      # into C later  
      CM = reshape(copy(CT), (props.dleft, props.dright))
    end
  else
    if Ctrans(props)
      CM = transpose(reshape(CT, (props.dright, props.dleft)))
    else
      CM = reshape(CT, (props.dleft, props.dright))
    end
  end

  #tC = similar(CM)
  #_gemm!(tA, tB, El(α), AM, BM, El(β), CM)
  mul!(CM, AM, BM, El(α), El(β))

  if props.permuteC
    Cr = reshape(CM, props.newCrange)
    # TODO: use invperm(pC) here?
    #@timeit_debug timer "_contract!: permutedims C" begin
    @strided CT .= permutedims(Cr, props.PC)
    #end # @timeit
  end

  return CT
end
