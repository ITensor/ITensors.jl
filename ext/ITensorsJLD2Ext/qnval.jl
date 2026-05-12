using ITensors: ITensors, QNVal, modulus
using JLD2: JLD2

struct SerializedQNVal
    version::Int
    name::String
    val::Int
    modulus::Int
end

JLD2.writeas(::Type{QNVal}) = SerializedQNVal

function JLD2.wconvert(::Type{SerializedQNVal}, qv::QNVal)
    version = 1
    return SerializedQNVal(
        version,
        String(ITensors.name(qv)),
        ITensors.val(qv),
        modulus(qv)
    )
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types. Unrelated to loading
# legacy (pre-extension) files, which don't have a :written_type attribute and are handled
# by JLD2 directly without calling rconvert.
JLD2.rconvert(::Type{QNVal}, qv::QNVal) = qv

JLD2.rconvert(::Type{QNVal}, s) = QNVal(s.name, s.val, s.modulus)
