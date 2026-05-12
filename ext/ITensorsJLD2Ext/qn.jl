using ITensors: QN, nactive
using JLD2: JLD2

struct SerializedQN
    version::Int
    qnvals::Vector{SerializedQNVal}
end

JLD2.writeas(::Type{QN}) = SerializedQN

function JLD2.wconvert(::Type{SerializedQN}, qn::QN)
    version = 1
    qnvals = SerializedQNVal[JLD2.wconvert(SerializedQNVal, qn[i]) for i in 1:nactive(qn)]
    return SerializedQN(version, qnvals)
end

JLD2.rconvert(::Type{QN}, qn::QN) = qn

function JLD2.rconvert(::Type{QN}, s)
    return QN([(qv.name, qv.val, qv.modulus) for qv in s.qnvals]...)
end
