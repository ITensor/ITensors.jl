using JLD2: JLD2
using NDTensors: EmptyStorage, SerializedEmptyStorage

JLD2.writeas(::Type{<:EmptyStorage{T}}) where {T} = SerializedEmptyStorage{T}

function JLD2.wconvert(::Type{SerializedEmptyStorage{T}}, e::EmptyStorage{T}) where {T}
    return SerializedEmptyStorage{T}(UInt32(1))
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types.
# TODO: Remove this idempotent method once the JLD2 bug is fixed.
JLD2.rconvert(::Type{S}, e::S) where {T, S <: EmptyStorage{T}} = e

JLD2.rconvert(::Type{S}, s) where {T, S <: EmptyStorage{T}} = S()
