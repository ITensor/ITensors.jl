function matrix_contract(A::ITensor, B::ITensor)
  A_inds = inds(A)
  B_inds = inds(B)
  NA = ndims(A)
  NB = ndims(B)

  labelsA, labelsB = compute_contraction_labels(inds(A), inds(B))
  labelsoutput_tensor = NDTensors.contract_labels(labelsA, labelsB)

  output_tensor = NDTensors.contraction_output(
    NDTensors.tensor(A), labelsA, NDTensors.tensor(B), labelsB, labelsoutput_tensor
  )
  NC = ndims(output_tensor)
  props = NDTensors.ContractionProperties(labelsA, labelsB, labelsoutput_tensor)
  NDTensors.compute_contraction_properties!(props, A, B, output_tensor)

  dmid = Index(props.dmid)
  dleft = Index(props.dleft)
  dright = Index(props.dright)

  vA = NDTensors.data(storage(A))
  vB = NDTensors.data(storage(B))

  if props.permuteA
    pA = NTuple{NA,Int}(props.PA)
    #@timeit_debug timer "_contract!: permutedims A" begin
    Ap = permutedims(NDTensors.tensor(A), pA)
    #end # @timeit
    AM = ITensor(storage(Ap), (dmid, dleft))
  else
    #A doesn't have to be permuted
    # Don't do the transpose here, do it later in the regular contract function
    if NDTensors.Atrans(props)
      #   println("transposing")
      #   tranA = transpose(NDTensors.ReshapedArray(NDTensors.data(storage(A)), (props.dmid, props.dleft), ()))
      AM = itensor(storage(A), (dmid, dleft))
      #   @show AM
    else
      AM = itensor(storage(A), (dleft, dmid))
    end
  end

  if props.permuteB
    pB = NTuple{NB,Int}(props.PB)
    #@timeit_debug timer "_contract!: permutedims B" begin
    Bp = permutedims(NDTensors.tensor(B), pB)
    #end # @timeit
    BM = itensor(storage(Bp), (dright, dmid))
  else
    if NDTensors.Btrans(props)
      BM = itensor(storage(B), (dright, dmid))
    else
      BM = itensor(storage(B), (dmid, dright))
    end
  end

  output_tensor = setinds!(_contract(AM, BM), inds(output_tensor))
  return output_tensor
end
