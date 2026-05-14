using JLD2: JLD2
using NDTensors:
    NDTensors, Diag, NonuniformDiag, SerializedDiag, SerializedUniformDiag, UniformDiag

# --- Diag (nonuniform) ---

JLD2.writeas(::Type{<:NonuniformDiag{T}}) where {T} = SerializedDiag{T}

function JLD2.wconvert(::Type{SerializedDiag{T}}, d::NonuniformDiag{T}) where {T}
    version = 1
    return SerializedDiag{T}(version, convert(Vector{T}, NDTensors.data(d)))
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types.
# TODO: Remove this idempotent method once the JLD2 bug is fixed.
JLD2.rconvert(::Type{S}, d::S) where {T, S <: NonuniformDiag{T}} = d

# Uses Diag{T} constructor because S(...) doesn't exist in NDTensors.
function JLD2.rconvert(::Type{S}, s) where {T, S <: NonuniformDiag{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    return Diag{T}(s.data)::S
end

# --- Diag (uniform) ---

JLD2.writeas(::Type{<:UniformDiag{T}}) where {T} = SerializedUniformDiag{T}

function JLD2.wconvert(::Type{SerializedUniformDiag{T}}, d::UniformDiag{T}) where {T}
    version = 1
    return SerializedUniformDiag{T}(version, convert(T, NDTensors.data(d)))
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types.
# TODO: Remove this idempotent method once the JLD2 bug is fixed.
JLD2.rconvert(::Type{S}, d::S) where {T, S <: UniformDiag{T}} = d

# Uses unparameterized Diag constructor because S(...) doesn't exist in NDTensors.
function JLD2.rconvert(::Type{S}, s) where {T, S <: UniformDiag{T}}
    s.value isa T ||
        throw(ArgumentError("value type mismatch: expected $T, got $(typeof(s.value))"))
    return Diag(s.value)::S
end
