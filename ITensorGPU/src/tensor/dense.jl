function contract!!(
  R::DenseTensor{<:Number,NR},
  labelsR::NTuple{NR},
  T1::DenseTensor{<:Number,N1},
  labelsT1::NTuple{N1},
  T2::DenseTensor{<:Number,N2},
  labelsT2::NTuple{N2},
  α::Number=1,
  β::Number=0,
) where {NR,N1,N2}
  if N1 == 0
    (α ≠ 1 || β ≠ 0) &&
      error("contract!! not yet implemented for scalar ITensor with non-trivial α and β")
    # TODO: replace with an add! function?
    # What about doing `R .= T1[] .* PermutedDimsArray(T2,perm)`?
    perm = getperm(labelsR, labelsT2)
    R = permutedims!!(R, T2, perm, (r, t2) -> T1[] * t2)
  elseif N2 == 0
    (α ≠ 1 || β ≠ 0) &&
      error("contract!! not yet implemented for scalar ITensor with non-trivial α and β")
    perm = getperm(labelsR, labelsT1)
    R = permutedims!!(R, T1, perm, (r, t1) -> T2[] * t1)
  elseif N1 + N2 == NR
    (α ≠ 1 || β ≠ 0) && error(
      "contract!! not yet implemented for outer product tensor contraction with non-trivial α and β",
    )
    # TODO: permute T1 and T2 appropriately first (can be more efficient
    # then permuting the result of T1⊗T2)
    # TODO: implement the in-place version directly
    R = outer!!(R, T1, T2)
    labelsRp = (labelsT1..., labelsT2...)
    perm = getperm(labelsR, labelsRp)
    if !is_trivial_permutation(perm)
      R = permutedims!!(R, copy(R), perm)
    end
  else
    #if dim(T1) > 2^13 && dim(T2) > 2^13 
    #    R = _big_contract!!(R,labelsR,T1,labelsT1,T2,labelsT2, α, β)
    #else
    if α ≠ 1 || β ≠ 0
      R = _contract!!(R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
    else
      R = _contract!!(R, labelsR, T1, labelsT1, T2, labelsT2)
    end
    #end
  end
  return R
end

function _big_contract!!(
  R::DenseTensor{<:Number,NR},
  labelsR,
  T1::DenseTensor{ElT1,N1},
  labelsT1,
  T2::DenseTensor{ElT2,N2},
  labelsT2,
  α::Number=1,
  β::Number=0,
) where {ElT1,ElT2,N1,N2,NR}
  props = ContractionProperties(labelsT1, labelsT2, labelsR)
  compute_contraction_properties!(props, T1, T2, R)
  _big_contract!(R, T1, T2, props, α, β)
  #val, t, _ = @timed _blasmg_contract!(R,T1,T2,props,α,β)
  return R
end

function _big_contract!(
  CT::DenseTensor{El,NC},
  AT::DenseTensor{El,NA},
  BT::DenseTensor{El,NB},
  props::ContractionProperties,
  α::Number=one(El),
  β::Number=zero(El),
) where {El,NC,NA,NB}
  Ainds = inds(AT)
  Adims = dims(Ainds)
  Binds = inds(BT)
  Bdims = dims(Binds)
  Cinds = inds(CT)
  Cdims = dims(Cinds)
  Adata = reshape(data(store(AT)), Adims)
  Bdata = reshape(data(store(BT)), Bdims)
  Cdata = reshape(data(store(CT)), Cdims)
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
    ctainds[ii] = findfirst(x -> x == ia, ind_dict)
  end
  for (ii, ib) in enumerate(Binds)
    ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
  end
  for (ii, ic) in enumerate(Cinds)
    ctcinds[ii] = findfirst(x -> x == ic, ind_dict)
  end
  id_op = cuTENSOR.CUTENSOR_OP_IDENTITY
  dict_key = ""
  for cc in zip(ctcinds, Cdims)
    dict_key *= string(cc[1]) * "," * string(cc[2]) * ","
  end
  for aa in zip(ctainds, Adims)
    dict_key *= string(aa[1]) * "," * string(aa[2]) * ","
  end
  for bb in zip(ctbinds, Bdims)
    dict_key *= string(bb[1]) * "," * string(bb[2]) * ","
  end
  #=synchronize()
  if haskey(ENV, "CUTENSOR_AUTOTUNE") && tryparse(Int, ENV["CUTENSOR_AUTOTUNE"]) == 1
      if haskey(ContractionPlans, dict_key)
          dict_val = ContractionPlans[dict_key]
          algo  = dict_val
          Cdata = cuTENSOR.contraction!(α, Adata, Vector{Char}(ctainds), id_op, Bdata, Vector{Char}(ctbinds), id_op, β, Cdata, Vector{Char}(ctcinds), id_op, id_op; algo=algo)
          synchronize()
      else
          # loop through all algos
          # pick the fastest one
          # store that plan!
          best_time = 1e6
          best_plan = nothing
          best_algo = nothing
          max_algos = Ref{Int32}(C_NULL)
          cuTENSOR.cutensorContractionMaxAlgos(max_algos)
          # fix once the other options are documented
          #algos = collect(Cint(cuTENSOR.CUTENSOR_ALGO_GETT):Cint(max_algos[] - 1))
          algos = collect(Cint(cuTENSOR.CUTENSOR_ALGO_GETT):Cint(-1))
          for algo in reverse(algos)
              try
                  Cdata, this_time, bytes, gctime, memallocs = @timed cuTENSOR.contraction!(α, Adata, Vector{Char}(ctainds), id_op, Bdata, Vector{Char}(ctbinds), id_op, β, Cdata, Vector{Char}(ctcinds), id_op, id_op; algo=cuTENSOR.cutensorAlgo_t(algo))
                  synchronize()
                  if this_time < best_time
                      best_time = this_time
                      best_algo = cuTENSOR.cutensorAlgo_t(algo)
                  end
              catch err
                  @warn "Algorithm $algo not supported"
              end
          end
          ContractionPlans[dict_key] = best_algo
      end
  else
  =#
  Cdata .= zero(eltype(Cdata))
  #@show size(Adata)
  #@show size(Bdata)
  #@show size(Cdata)
  @assert !any(isnan.(Adata))
  @assert !any(isnan.(Bdata))
  @assert !any(isnan.(Cdata))
  #@show ctainds
  #@show ctbinds
  #@show ctcinds
  #flush(stdout)
  CUDA.Mem.pin(Adata)
  CUDA.Mem.pin(Bdata)
  CUDA.Mem.pin(Cdata)
  synchronize()
  #AC = CuArray(Adata)
  #BC = CuArray(Bdata)
  #CC = CuArray(Cdata)
  @assert !any(isnan.(Adata))
  @assert !any(isnan.(Bdata))
  @assert !any(isnan.(Cdata))
  #@assert !any(isnan.(AC))
  #@assert !any(isnan.(BC))
  #@assert !any(isnan.(CC))
  #CC = cuTENSOR.contraction!(α, AC, ctainds, id_op, BC, ctbinds, id_op, β, CC, ctcinds, id_op, id_op)
  #synchronize()
  #@assert !any(isnan.(AC))
  #@assert !any(isnan.(BC))
  #@assert !any(isnan.(CC))
  Cdata = cuTENSOR.contraction!(
    α, Adata, ctainds, id_op, Bdata, ctbinds, id_op, β, Cdata, ctcinds, id_op, id_op
  )
  synchronize()
  #end
  #CCh = collect(CC)
  #@assert !any(isnan.(CCh))
  #Cdata .= CCh
  @assert !any(isnan.(Adata))
  @assert !any(isnan.(Bdata))
  @assert !any(isnan.(Cdata))
  return parent(Cdata)
end

function _blasmg_contract!(
  CT::DenseTensor{El,NC},
  AT::DenseTensor{El,NA},
  BT::DenseTensor{El,NB},
  props::ContractionProperties,
  α::Number=one(El),
  β::Number=zero(El),
) where {El,NC,NA,NB}
  # TODO: directly use Tensor instead of Array
  C = array(CT)
  A = array(AT)
  B = array(BT)

  tA = 'N'
  if props.permuteA
    pA = NTuple{NA,Int}(props.PA)
    @strided Ap = permutedims(A, pA)
    AM = reshape(Ap, props.dmid, props.dleft)
    tA = 'T'
  else
    #A doesn't have to be permuted
    if Atrans(props)
      AM = reshape(A, props.dmid, props.dleft)
      tA = 'T'
    else
      AM = reshape(A, props.dleft, props.dmid)
    end
  end

  tB = 'N'
  if props.permuteB
    pB = NTuple{NB,Int}(props.PB)
    @strided Bp = permutedims(B, pB)
    BM = reshape(Bp, props.dmid, props.dright)
  else
    if Btrans(props)
      BM = reshape(B, props.dright, props.dmid)
      tB = 'T'
    else
      BM = reshape(B, props.dmid, props.dright)
    end
  end

  # TODO: this logic may be wrong
  if props.permuteC
    # Need to copy here since we will be permuting
    # into C later
    CM = reshape(copy(C), props.dleft, props.dright)
  else
    if Ctrans(props)
      CM = reshape(C, props.dleft, props.dright)
      (AM, BM) = (BM, AM)
      if tA == tB
        tA = tB = (tA == 'T' ? 'N' : 'T')
      end
    else
      CM = reshape(C, props.dleft, props.dright)
    end
  end

  if length(AM) > 4096 && length(BM) > 4096 && length(CM) > 4096
    CM = CUBLASMG.mg_gemm!(
      tA,
      tB,
      El(α),
      AM,
      BM,
      El(β),
      CM;
      devs=devs[],
      dev_rows=dev_rows[],
      dev_cols=dev_cols[],
    )
  else
    BLAS.gemm!(tA, tB, El(α), AM, BM, El(β), CM)
  end

  if props.permuteC
    pC = NTuple{NC,Int}(props.PC)
    Cr = reshape(CM, props.newCrange...)
    @strided C .= permutedims(Cr, pC)
  end
  return C
end
