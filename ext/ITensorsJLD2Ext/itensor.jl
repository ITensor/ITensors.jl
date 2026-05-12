using ITensors: ITensor, inds, itensor, storage
using JLD2: JLD2

struct SerializedITensor
    version::Int
    storage::Any
    inds::Vector
end

JLD2.writeas(::Type{ITensor}) = SerializedITensor

function JLD2.wconvert(::Type{SerializedITensor}, it::ITensor)
    version = 1
    return SerializedITensor(version, storage(it), collect(inds(it)))
end

JLD2.rconvert(::Type{ITensor}, it::ITensor) = it

JLD2.rconvert(::Type{ITensor}, s) = itensor(s.storage, Tuple(s.inds))
