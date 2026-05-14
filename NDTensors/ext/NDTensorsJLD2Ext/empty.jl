using JLD2: JLD2
using NDTensors: EmptyStorage, SerializedEmptyStorage

JLD2.writeas(::Type{<:EmptyStorage{T}}) where {T} = SerializedEmptyStorage{T}

function JLD2.wconvert(::Type{SerializedEmptyStorage{T}}, e::EmptyStorage{T}) where {T}
    version = 1
    return SerializedEmptyStorage{T}(version, T)
end

JLD2.rconvert(::Type{S}, e::S) where {T, S <: EmptyStorage{T}} = e

function JLD2.rconvert(::Type{S}, s) where {T, S <: EmptyStorage{T}}
    T === s.eltype || throw(ArgumentError("eltype mismatch: expected $T, got $(s.eltype)"))
    return S()
end
