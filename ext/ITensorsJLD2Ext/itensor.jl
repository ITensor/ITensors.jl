using ITensors: ITensor, SerializedITensor, inds, itensor, storage
using JLD2: JLD2

JLD2.writeas(::Type{ITensor}) = SerializedITensor

function JLD2.wconvert(::Type{SerializedITensor}, it::ITensor)
    version = UInt32(1)
    return SerializedITensor(version, storage(it), collect(inds(it)))
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types.
# TODO: Remove this idempotent method once the JLD2 bug is fixed.
JLD2.rconvert(::Type{ITensor}, it::ITensor) = it

JLD2.rconvert(::Type{ITensor}, s) = itensor(s.storage, Tuple(s.inds))
