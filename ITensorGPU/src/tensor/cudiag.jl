export CuDiag

const CuDiag{ElT,VecT} = Diag{ElT,VecT} where {VecT<:CuArray{ElT}}
const NonuniformCuDiagTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:CuDiag}

CuArray(D::NonuniformCuDiagTensor) = CuArray(dense(D))

function NDTensors.dense(T::NonuniformCuDiagTensor{ElT}) where {ElT}
  R_data = CUDA.zeros(ElT, dim(inds(T)))
  diag_inds = diagind(reshape(R_data, dims(inds(T))), 0)
  R_data[diag_inds] .= data(store(T))
  return Tensor(Dense(R_data), inds(T))
end

function Base.promote_rule(
  ::Type{Diag{ElT2,VecT2}}, ::Type{CuDense}
) where {ElT2,VecT2<:Number}
  return promote_type(DenseT1, ElT2)
end

function Base.promote_rule(
  ::Type{<:Tensor{ElT1,N1,StoreT1}}, ::Type{<:Tensor{ElT2,N2,StoreT2}}
) where {ElT1,ElT2,N1,N2,StoreT1<:CuDense,StoreT2<:UniformDiag}
  ElT3 = promote_type(ElT1, ElT2)
  return Tensor{promote_type(ElT1, ElT2),N3,CuDense{ElT3,CuVector{ElT3}}} where {N3}
end
function Base.promote_rule(
  ::Type{<:Tensor{ElT1,N1,StoreT1}}, ::Type{<:Tensor{ElT2,N2,StoreT2}}
) where {ElT1,ElT2,N1,N2,StoreT1<:UniformDiag,StoreT2<:CuDense}
  ElT3 = promote_type(ElT1, ElT2)
  return Tensor{promote_type(ElT1, ElT2),N3,CuDense{ElT3,CuVector{ElT3}}} where {N3}
end

function Base.promote_rule(
  ::Type{<:UniformDiag{ElT1}}, ::Type{<:NonuniformDiag{ElT2,VecT2}}
) where {ElT1,ElT2,VecT2<:CuArray}
  ElT3 = promote_type(ElT1, ElT2)
  return NonuniformDiag{ElT3,CuVector{ElT3}}
end
function Base.promote_rule(
  ::Type{<:NonuniformDiag{ElT2,VecT2}}, ::Type{<:UniformDiag{ElT1}}
) where {ElT1,ElT2,VecT2<:CuArray}
  ElT3 = promote_type(ElT1, ElT2)
  return NonuniformDiag{ElT3,CuVector{ElT3}}
end
function Base.promote_rule(
  ::Type{DenseT1}, ::Type{Diag{ElT2,VecT2}}
) where {DenseT1<:CuDense,ElT2,VecT2<:Number}
  return promote_type(DenseT1, ElT2)
end

function contraction_output_type(
  TensorT1::Type{<:NonuniformCuDiagTensor}, TensorT2::Type{<:CuDenseTensor}, IndsR
)
  return similartype(promote_type(TensorT1, TensorT2), IndsR)
end
function contraction_output_type(
  TensorT1::Type{<:CuDenseTensor}, TensorT2::Type{<:NonuniformCuDiagTensor}, IndsR
)
  return contraction_output_type(TensorT2, TensorT1, IndsR)
end

function contraction_output_type(
  TensorT1::Type{<:DiagTensor{<:Number,<:CuDiag}},
  TensorT2::Type{<:DiagTensor{<:Number,<:CuDiag}},
  IndsR::Type,
)
  return similartype(promote_type(TensorT1, TensorT2), IndsR)
end

function contraction_output_type(
  TensorT1::Type{<:UniformDiagTensor},
  TensorT2::Type{<:DiagTensor{<:Number,<:CuDiag}},
  IndsR::Type,
)
  return similartype(promote_type(TensorT1, TensorT2), IndsR)
end
function contraction_output_type(
  TensorT1::Type{<:DiagTensor{<:Number,<:CuDiag}},
  TensorT2::Type{<:UniformDiagTensor},
  IndsR::Type,
)
  return contraction_output_type(TensorT2, TensorT1, IndsR)
end

function zero_contraction_output(
  T1::UniformDiagTensor{ElT1,N1}, T2::CuDenseTensor{ElT2,N2}, indsR::IndsR
) where {ElT1,N1,ElT2,N2,IndsR}
  ElT3 = promote_type(ElT1, ElT2)
  dat = CUDA.zeros(ElT3, dim(indsR))
  return Tensor(Dense(dat), indsR)
end
function zero_contraction_output(
  T2::CuDenseTensor{ElT2,N2}, T1::UniformDiagTensor{ElT1,N1}, indsR::IndsR
) where {ElT1,N1,ElT2,N2,IndsR}
  return zero_contraction_output(T1, T2, indsR)
end

function zero_contraction_output(
  T1::TensorT1, T2::TensorT2, indsR
) where {TensorT1<:NonuniformDiagTensor,TensorT2<:NonuniformCuDiagTensor}
  ElT3 = promote_type(eltype(TensorT1), eltype(TensorT2))
  dat = CUDA.zeros(ElT3, length(data(store(T2))))
  return Tensor(Diag(dat), indsR)
end

function zero_contraction_output(
  T1::TensorT1, T2::TensorT2, indsR
) where {TensorT1<:UniformDiagTensor,TensorT2<:NonuniformCuDiagTensor}
  ElT3 = promote_type(eltype(TensorT1), eltype(TensorT2))
  dat = CUDA.zeros(ElT3, length(data(store(T2))))
  return Tensor(Diag(dat), indsR)
end
function zero_contraction_output(
  T1::TensorT1, T2::TensorT2, indsR
) where {TensorT2<:UniformDiagTensor,TensorT1<:NonuniformCuDiagTensor}
  return zero_contraction_output(T2, T1, indsR)
end

function zero_contraction_output(
  T1::TensorT1, T2::TensorT2, indsR
) where {TensorT1<:DiagTensor,TensorT2<:CuDenseTensor}
  ElT3 = promote_type(eltype(TensorT1), eltype(TensorT2))
  dat = CUDA.zeros(ElT3, length(data(store(T2))))
  return Tensor(Dense(dat), indsR)
end
function zero_contraction_output(
  T1::TensorT1, T2::TensorT2, indsR
) where {TensorT2<:DiagTensor,TensorT1<:CuDenseTensor}
  return zero_contraction_output(T2, T1, indsR)
end

function contraction_output(
  T1::UniformDiagTensor, T2::DiagTensor{Elt2,<:CuDiag}, indsR
) where {Elt2}
  return zero_contraction_output(T1, T2, indsR)
end
function contraction_output(
  T1::DiagTensor{Elt1,<:CuDiag}, T2::UniformDiagTensor, indsR
) where {Elt1}
  return contraction_output(T2, T1, indsR)
end

function contract!(
  C::CuDenseTensor{<:Number,NC},
  Clabels,
  A::UniformDiagTensor{<:Number,NA},
  Alabels,
  B::CuDenseTensor{<:Number,NB},
  Blabels,
) where {NC,NA,NB}
  Bstore = data(store(B))
  Astore = data(store(A))
  Cstore = data(store(C))
  return copyto!(Cstore, Astore .* Bstore)
end

function contract!(
  C::CuDenseTensor{<:Number,NC},
  Clabels,
  A::CuDenseTensor{<:Number,NA},
  Alabels,
  B::UniformDiagTensor{<:Number,NB},
  Blabels,
) where {NC,NA,NB}
  return contract!(C, Clabels, B, Blabels, A, Alabels)
end

function contract!(
  C::NonuniformCuDiagTensor{EltC,NC,IndsC},
  Clabels,
  A::UniformDiagTensor{EltA,NA,IndsA},
  Alabels,
  B::NonuniformCuDiagTensor{EltB,NB,IndsB},
  Blabels,
) where {EltC<:Number,EltB<:Number,EltA<:Number,NC,NB,NA,IndsA,IndsB,IndsC}
  Bstore = data(store(B))
  Astore = data(store(A))
  Cstore = data(store(C))
  return copyto!(Cstore, Astore .* Bstore)
end

function contract!(
  C::NonuniformCuDiagTensor{EltC,NC,IndsC},
  Clabels,
  B::NonuniformCuDiagTensor{EltB,NB,IndsB},
  Blabels,
  A::UniformDiagTensor{EltA,NA,IndsA},
  Alabels,
) where {EltC<:Number,EltB<:Number,EltA<:Number,NC,NB,NA,IndsA,IndsB,IndsC}
  Bstore = data(store(B))
  Astore = data(store(A))
  Cstore = data(store(C))
  return copyto!(Cstore, Astore .* Bstore)
end

function contract!(
  C::NonuniformCuDiagTensor{EltC,NC,IndsC},
  Clabels,
  B::NonuniformCuDiagTensor{EltB,NB,IndsB},
  Blabels,
  A::NonuniformCuDiagTensor{EltA,NA,IndsA},
  Alabels,
) where {EltC<:Number,EltB<:Number,EltA<:Number,NC,NB,NA,IndsA,IndsB,IndsC}
  Bstore = data(store(B))
  Astore = data(store(A))
  Cstore = data(store(C))
  return copyto!(Cstore, Astore .* Bstore)
end

# Dense * NonuniformCuDiag
function contract!(
  C::CuDenseTensor, Clabels, A::NonuniformCuDiagTensor, Alabels, B::CuDenseTensor, Blabels
)
  Astore = data(store(A))
  newAstore = CUDA.zeros(eltype(A), dims(inds(A))[1], dims(inds(A))[2])
  adi = diagind(newAstore, 0)
  newAstore[adi] .= Astore[:]
  newA = Tensor(Dense(vec(newAstore)), inds(A))
  return contract!(C, Clabels, newA, Alabels, B, Blabels)
end

function contract!(
  C::CuDenseTensor, Clabels, A::CuDenseTensor, Alabels, B::NonuniformCuDiagTensor, Blabels
)
  return contract!(C, Clabels, B, Blabels, A, Alabels)
end
