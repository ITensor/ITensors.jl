using ITensors: Arrow, Index, QNIndex, SerializedIndex, SerializedQNSpace, SerializedTagSet,
    TagSet, dir, id, plev, space, tags
using JLD2: JLD2

# Two explicit declarations rather than the natural-looking
# `JLD2.writeas(::Type{<:Index{S}}) where {S} = SerializedIndex{JLD2.writeas(S)}`:
# the parametric form also covers `Index{SerializedQNSpace}`, which collapses to the same
# target as `Index{Vector{Pair{QN, Int}}}` and confuses JLD2's inverse-lookup on read.
JLD2.writeas(::Type{<:Index{Int}}) = SerializedIndex{Int}
JLD2.writeas(::Type{<:QNIndex}) = SerializedIndex{SerializedQNSpace}

function JLD2.wconvert(
        ::Type{SerializedIndex{ST}}, i::Index{S}
    ) where {ST, S}
    version = 1
    sp = ST === S ? space(i) : JLD2.wconvert(ST, space(i))
    return SerializedIndex{ST}(
        version, id(i), sp, Int(dir(i)),
        JLD2.wconvert(SerializedTagSet, tags(i)), plev(i)
    )
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types.
# TODO: Remove this idempotent method once the JLD2 bug is fixed.
JLD2.rconvert(::Type{I}, i::I) where {T, I <: Index{T}} = i

function JLD2.rconvert(::Type{I}, s) where {T, I <: Index{T}}
    sp = T === typeof(s.space) ? s.space : JLD2.rconvert(T, s.space)
    return I(s.id, sp, Arrow(s.dir), JLD2.rconvert(TagSet, s.tags), s.plev)
end
