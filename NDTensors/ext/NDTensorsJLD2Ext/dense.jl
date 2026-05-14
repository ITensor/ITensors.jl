using JLD2: JLD2
using NDTensors: NDTensors, Dense, SerializedDense

JLD2.writeas(::Type{<:Dense{T}}) where {T} = SerializedDense{T}

function JLD2.wconvert(::Type{SerializedDense{T}}, d::Dense{T}) where {T}
    version = 1
    return SerializedDense{T}(version, convert(Vector{T}, NDTensors.data(d)))
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types. Unrelated to loading
# legacy (pre-extension) files, which don't have a :written_type attribute and are handled
# by JLD2 directly without calling rconvert.
JLD2.rconvert(::Type{S}, d::S) where {T, S <: Dense{T}} = d

function JLD2.rconvert(::Type{S}, s) where {T, S <: Dense{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    return S(s.data)
end
