# Version of contraction where output storage is empty
function contract!!(R::EmptyTensor, labelsR, T1::Tensor, labelsT1, T2::Tensor, labelsT2)
  RR = contract(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

# When one of the tensors is empty, return an empty
# tensor.
# XXX: make sure `R` is actually correct!
function contract!!(
  R::EmptyTensor, labelsR, T1::EmptyTensor, labelsT1, T2::Tensor, labelsT2
)
  return R
end

# When one of the tensors is empty, return an empty
# tensor.
# XXX: make sure `R` is actually correct!
function contract!!(
  R::EmptyTensor, labelsR, T1::Tensor, labelsT1, T2::EmptyTensor, labelsT2
)
  return R
end

function contract!!(
  R::EmptyTensor, labelsR, T1::EmptyTensor, labelsT1, T2::EmptyTensor, labelsT2
)
  return R
end

# For ambiguity with versions in combiner.jl
function contract!!(
  R::EmptyTensor, labelsR, T1::CombinerTensor, labelsT1, T2::Tensor, labelsT2
)
  RR = contract(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

# For ambiguity with versions in combiner.jl
function contract!!(
  R::EmptyTensor, labelsR, T1::Tensor, labelsT1, T2::CombinerTensor, labelsT2
)
  RR = contract(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

# For ambiguity with versions in combiner.jl
function contract!!(
  R::EmptyTensor, labelsR, T1::EmptyTensor, labelsT1, T2::CombinerTensor, labelsT2
)
  RR = contraction_output(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

function contraction_output(T1::EmptyTensor, T2::EmptyTensor, indsR::Tuple)
  fulltypeR = contraction_output_type(fulltype(T1), fulltype(T2), indsR)
  storagetypeR = storagetype(fulltypeR)
  emptystoragetypeR = emptytype(storagetypeR)
  return Tensor(emptystoragetypeR(), indsR)
end

function contraction_output(T1::Tensor, T2::EmptyTensor, indsR)
  fulltypeR = contraction_output_type(typeof(T1), fulltype(T2), indsR)
  storagetypeR = storagetype(fulltypeR)
  emptystoragetypeR = emptytype(storagetypeR)
  return Tensor(emptystoragetypeR(), indsR)
end

function contraction_output(T1::EmptyTensor, T2::Tensor, indsR)
  fulltypeR = contraction_output_type(fulltype(T1), typeof(T2), indsR)
  storagetypeR = storagetype(fulltypeR)
  emptystoragetypeR = emptytype(storagetypeR)
  return Tensor(emptystoragetypeR(), indsR)
end
