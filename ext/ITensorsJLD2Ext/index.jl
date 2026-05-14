using ITensors: Arrow, Index, QNIndex, SerializedIndex, SerializedQNSpace, SerializedTagSet,
    TagSet, dir, id, plev, space, tags
using JLD2: JLD2

JLD2.writeas(::Type{<:Index{Int}}) = SerializedIndex{Int}
JLD2.writeas(::Type{<:QNIndex}) = SerializedIndex{SerializedQNSpace}

function JLD2.wconvert(::Type{<:SerializedIndex}, i::Index)
    version = 1
    return SerializedIndex(
        version, id(i), _wconvert_space(space(i)), Int(dir(i)),
        JLD2.wconvert(SerializedTagSet, tags(i)), plev(i)
    )
end

JLD2.rconvert(::Type{I}, i::I) where {T, I <: Index{T}} = i

function JLD2.rconvert(::Type{I}, s) where {T, I <: Index{T}}
    return I(
        s.id, _rconvert_space(s.space), Arrow(s.dir), JLD2.rconvert(TagSet, s.tags), s.plev
    )
end
