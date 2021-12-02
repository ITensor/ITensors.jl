function cuITensor(::Type{T},inds::IndexSet) where {T<:Number}
    return ITensor(Dense{float(T)}(CUDA.zeros(float(T),dim(inds))), inds)
end
cuITensor(::Type{T},inds::Index...) where {T<:Number} = cuITensor(T,IndexSet(inds...))

cuITensor(is::IndexSet)   = cuITensor(Float64,is)
cuITensor(inds::Index...) = cuITensor(IndexSet(inds...))

cuITensor() = ITensor()
function cuITensor(x::S, inds::IndexSet{N}) where {S<:Number, N}
    dat = CuVector{float(S)}(undef, dim(inds))
    fill!(dat, float(x))
    ITensor(Dense{S}(dat), inds)
end
cuITensor(x::S, inds::Index...) where {S<:Number} = cuITensor(x,IndexSet(inds...))

#TODO: check that the size of the Array matches the Index dimensions
function cuITensor(A::Array{S}, inds) where {S<:Number}
    return ITensor(Dense(CuArray{S}(A)), inds)
end
function cuITensor(A::CuArray{S}, inds::IndexSet) where {S<:Number}
    return ITensor(Dense(A), inds)
end
cuITensor(A::Array{S},   inds::Index...) where {S<:Number} = cuITensor(A,IndexSet(inds...))
cuITensor(A::CuArray{S}, inds::Index...) where {S<:Number} = cuITensor(A,IndexSet(inds...))
cuITensor(A::ITensor) = storage(tensor(A)) isa ITensors.EmptyStorage ? cuITensor(zero(eltype(storage(tensor(A)))), inds(A)...) : cuITensor(data(tensor(A)), inds(A)...)

cu(A::ITensor) = cuITensor(A)

function cpu(A::ITensor)
    typeof(data(storage(A))) <: CuArray && return ITensor(cpu(storage(A)), inds(A))    
    return A
end

function randomCuITensor(::Type{S},inds::Indices) where {S<:Real}
  T = cuITensor(S,inds)
  randn!(T)
  return T
end
function randomCuITensor(::Type{S},inds::Indices) where {S<:Complex}
  Tr = cuITensor(real(S),inds)
  Ti = cuITensor(real(S),inds)
  randn!(Tr)
  randn!(Ti)
  return complex(Tr) + im * Ti
end
randomCuITensor(::Type{S},inds::Index...) where {S<:Number} = randomCuITensor(S,IndexSet(inds...))
randomCuITensor(inds::IndexSet) = randomCuITensor(Float64,inds)
randomCuITensor(inds::Index...) = randomCuITensor(Float64,IndexSet(inds...))

CuArray(T::ITensor) = CuArray(tensor(T))

function CuArray{ElT, N}(T::ITensor,
                         is::Vararg{Index, N}) where {ElT, N}
  ndims(T) != N && throw(DimensionMismatch("cannot convert an $(ndims(T)) dimensional ITensor to an $N-dimensional CuArray."))
  TT = tensor(permute(T, is...; allow_alias=true))
  return CuArray{ElT, N}(TT)::CuArray{ElT, N}
end

function CuArray{ElT}(T::ITensor, is::Vararg{Index, N}) where {ElT, N}
    return CuArray{ElT, N}(T, is...)
end

function CuArray(T::ITensor, is::Vararg{Index, N}) where {N}
  return CuArray{eltype(T), N}(T, is...)::CuArray{<:Number, N}
end

CUDA.CuMatrix(A::ITensor) = CuArray(A)

function CuVector(A::ITensor)
  if ndims(A) != 1
    throw(DimensionMismatch("Vector() expected a 1-index ITensor"))
  end
  return CuArray(A)
end

function CuMatrix(T::ITensor,i1::Index,i2::Index)
  ndims(T) != 2 && throw(DimensionMismatch("ITensor must be order 2 to convert to a Matrix"))
  return CuArray(T,i1,i2)
end
