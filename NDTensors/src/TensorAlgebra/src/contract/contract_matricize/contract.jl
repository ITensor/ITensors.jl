function contract!(alg::Algorithm"matricize", a_dest::AbstractArray, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β; plan=nothing)
  contract!(alg, plan, a_dest, labels_dest, a1, labels1, a2, labels2, α, β)
  return a_dest
end

function contract!(alg::Algorithm"matricize", plan::Nothing, a_dest::AbstractArray, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β)
  plan = ContractionProperties(labels1, labels2, labels_dest)
  compute_contraction_properties!(plan, a1, a2, a_dest)
  contract!(alg, plan, a_dest, labels_dest, a1, labels1, a2, labels2, α, β)
  return a_dest
end

function contract!(alg::Algorithm"matricize", plan::ContractionProperties, a1::AbstractArray, labels1, a2::AbstractArray, labels2, α, β)
  @show alg
  @show plan
  @show labels1, labels2

  error("Not implemented")

  tA = 'N'
  if plan.permuteA
    Ap = permutedims(expose(AT), plan.PA)
    AM = transpose(reshape(Ap, (plan.dmid, plan.dleft)))
  else
    #A doesn't have to be permuted
    if Atrans(plan)
      AM = transpose(reshape(AT, (plan.dmid, plan.dleft)))
    else
      AM = reshape(AT, (plan.dleft, plan.dmid))
    end
  end

  tB = 'N'
  if plan.permuteB
    Bp = permutedims(expose(BT), plan.PB)
    BM = reshape(Bp, (plan.dmid, plan.dright))
  else
    if Btrans(plan)
      BM = transpose(reshape(BT, (plan.dright, plan.dmid)))
    else
      BM = reshape(BT, (plan.dmid, plan.dright))
    end
  end

  if plan.permuteC
    # if we are computing C = α * A B + β * C
    # we need to make sure C is permuted to the same 
    # ordering as A B which is the inverse of plan.PC
    if β ≠ 0
      CM = reshape(permutedims(expose(CT), invperm(plan.PC)), (plan.dleft, plan.dright))
    else
      # Need to copy here since we will be permuting
      # into C later  
      CM = reshape(copy(CT), (plan.dleft, plan.dright))
    end
  else
    if Ctrans(plan)
      CM = transpose(reshape(CT, (plan.dright, plan.dleft)))
    else
      CM = reshape(CT, (plan.dleft, plan.dright))
    end
  end

  ## CM = mul!!(CM, AM, BM, El(α), El(β))
  CM = mul!(CM, AM, BM, El(α), El(β))

  if plan.permuteC
    Cr = reshape(CM, plan.newCrange)
    CT .= permutedims(expose(Cr), plan.PC)
  end
  return CT
end
