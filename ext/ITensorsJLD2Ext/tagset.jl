using ITensors: TagSet
using JLD2: JLD2

struct SerializedTagSet
    version::Int
    tags::Vector{String}
end

JLD2.writeas(::Type{<:TagSet}) = SerializedTagSet

function JLD2.wconvert(::Type{SerializedTagSet}, ts::TagSet)
    version = 1
    return SerializedTagSet(version, String[String(ts[n]) for n in 1:length(ts)])
end

JLD2.rconvert(::Type{TagSet}, ts::TagSet) = ts

JLD2.rconvert(::Type{TagSet}, s) = TagSet(join(s.tags, ","))
