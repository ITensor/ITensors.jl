function cuITensor(::Type{T},inds::IndexSet) where {T<:Number}
    return ITensor(inds,Dense{T, CuVector{T}}(dim(inds)))
end
cuITensor(::Type{T},inds::Index...) where {T<:Number} = ITensor(T,IndexSet(inds...))

cuITensor(is::IndexSet) = cuITensor(Float64,is...)
cuITensor(inds::Index...) = cuITensor(IndexSet(inds...))

function cuITensor(x::S,inds::IndexSet) where {S<:Number}
    return ITensor(inds,Dense{S, CuVector{S}}(x,dim(inds)))
end
cuITensor(x::S,inds::Index...) where {S<:Number} = cuITensor(x,IndexSet(inds...))

#TODO: check that the size of the Array matches the Index dimensions
function cuITensor(A::Array{S},inds::IndexSet) where {S<:Number}
    return ITensor(inds,Dense{S, CuVector{S}}(CuArray{S}(A)))
end
function cuITensor(A::CuArray{S},inds::IndexSet) where {S<:Number}
    return ITensor(inds,Dense{S, CuVector{S}}(A))
end
cuITensor(A::Array{S},inds::Index...) where {S<:Number} = cuITensor(A,IndexSet(inds...))
cuITensor(A::CuArray{S},inds::Index...) where {S<:Number} = cuITensor(A,IndexSet(inds...))
cuITensor(A::ITensor) = cuITensor(A.store.data,A.inds)

cuITensor() = cuITensor(IndexSet())

function Base.collect(A::ITensor)
    if typeof(A.store.data) <: CuArray
        return ITensor(A.inds, collect(A.store))    
    else
        return A
    end
end

function randomCuITensor(::Type{S},inds::IndexSet) where {S<:Number}
  T = cuITensor(S,inds)
  randn!(T)
  return T
end
randomCuITensor(::Type{S},inds::Index...) where {S<:Number} = randomCuITensor(S,IndexSet(inds...))
randomCuITensor(inds::IndexSet) = randomCuITensor(Float64,inds)
randomCuITensor(inds::Index...) = randomCuITensor(Float64,IndexSet(inds...))

