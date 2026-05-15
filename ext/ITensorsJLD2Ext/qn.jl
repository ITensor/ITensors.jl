using ITensors: QN, SerializedQN, SerializedQNVal, nactive
using JLD2: JLD2

JLD2.writeas(::Type{QN}) = SerializedQN

function JLD2.wconvert(::Type{SerializedQN}, qn::QN)
    version = UInt32(1)
    qnvals = SerializedQNVal[JLD2.wconvert(SerializedQNVal, qn[i]) for i in 1:nactive(qn)]
    return SerializedQN(version, qnvals)
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types.
# TODO: Remove this idempotent method once the JLD2 bug is fixed.
JLD2.rconvert(::Type{QN}, qn::QN) = qn

function JLD2.rconvert(::Type{QN}, s)
    return QN([(qv.name, qv.val, qv.modulus) for qv in s.qnvals]...)
end
