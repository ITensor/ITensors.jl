using ITensors: ITensors, QNVal, SerializedQNVal, modulus
using JLD2: JLD2

JLD2.writeas(::Type{QNVal}) = SerializedQNVal

function JLD2.wconvert(::Type{SerializedQNVal}, qv::QNVal)
    version = UInt32(1)
    return SerializedQNVal(
        version,
        String(ITensors.name(qv)),
        ITensors.val(qv),
        modulus(qv)
    )
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types.
# TODO: Remove this idempotent method once the JLD2 bug is fixed.
JLD2.rconvert(::Type{QNVal}, qv::QNVal) = qv

JLD2.rconvert(::Type{QNVal}, s) = QNVal(s.name, s.val, s.modulus)
