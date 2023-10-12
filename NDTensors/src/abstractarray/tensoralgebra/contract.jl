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

# Non-trivial permutation
function _contract_scalar_perm!(
  Rᵃ::AbstractArray{ElR}, Tᵃ::AbstractArray, perm, α, β=zero(ElR)
) where {ElR}
  if iszero(β)
    if iszero(α)
      fill!(Rᵃ, 0)
    else
      Rᵃ = permutedims!!(Rᵃ, Tᵃ, perm, (r, t) -> α * t)
    end
  elseif isone(β)
    if iszero(α)
      # Rᵃ .= Rᵃ
      # No-op
    else
      Rᵃ = permutedims!!(Rᵃ, Tᵃ, perm, (r, t) -> r + α * t)
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
    Ap = permutedims(leaf_parenttype(AT), AT, props.PA)
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
    Bp = permutedims(leaf_parenttype(BT), BT, props.PB)
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
      CM = reshape(
        permutedims(leaf_parenttype(CT), CT, invperm(props.PC)), (props.dleft, props.dright)
      )
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
  CM = mul!!(CM, AM, BM, El(α), El(β)
  )

  if props.permuteC
    Cr = reshape(CM, props.newCrange)
    # TODO: use invperm(pC) here?
    #@timeit_debug timer "_contract!: permutedims C" begin 
    CT .= permutedims(leaf_parenttype(Cr), Cr, props.PC)
    #end # @timeit
  end

  return CT
end