using ITensors: SerializedTagSet, TagSet
using JLD2: JLD2

JLD2.writeas(::Type{<:TagSet}) = SerializedTagSet

function JLD2.wconvert(::Type{SerializedTagSet}, ts::TagSet)
    version = UInt32(1)
    return SerializedTagSet(version, String[String(ts[n]) for n in 1:length(ts)])
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types.
# TODO: Remove this idempotent method once the JLD2 bug is fixed.
JLD2.rconvert(::Type{TagSet}, ts::TagSet) = ts

JLD2.rconvert(::Type{TagSet}, s) = TagSet(join(s.tags, ","))
