using ..ITensors: ITensors, name, val
using ..SmallStrings: SmallString

struct QNVal
    name::SmallString
    val::Int
    modulus::Int
    function QNVal(name, v::Int, m::Int = 1)
        am = abs(m)
        if am > 1
            return new(SmallString(name), mod(v, am), m)
        end
        return new(SmallString(name), v, m)
    end
end

QNVal(v::Int, m::Int = 1) = QNVal("", v, m)
QNVal() = QNVal("", 0, 0)

ITensors.name(qv::QNVal) = qv.name
ITensors.val(qv::QNVal) = qv.val
modulus(qv::QNVal) = qv.modulus
isactive(qv::QNVal) = modulus(qv) != 0
Base.:(<)(qv1::QNVal, qv2::QNVal) = (name(qv1) < name(qv2))

function qn_mod(val::Int, modulus::Int)
    amod = abs(modulus)
    amod <= 1 && return val
    return mod(val, amod)
end

function Base.:(-)(qv::QNVal)
    return QNVal(name(qv), qn_mod(-val(qv), modulus(qv)), modulus(qv))
end

Base.zero(::Type{QNVal}) = QNVal()

Base.zero(qv::QNVal) = QNVal(name(qv), 0, modulus(qv))

Base.:(*)(dir::Arrow, qv::QNVal) = QNVal(name(qv), Int(dir) * val(qv), modulus(qv))

Base.:(*)(qv::QNVal, dir::Arrow) = (dir * qv)

function pm(qv1::QNVal, qv2::QNVal, fac::Int)
    if name(qv1) != name(qv2)
        error("Cannot add QNVals with different names \"$(name(qv1))\", \"$(name(qv2))\"")
    end
    if modulus(qv1) != modulus(qv2)
        error(
            "QNVals with matching name \"$(name(qv1))\" cannot have different modulus values "
        )
    end
    m1 = modulus(qv1)
    if m1 == 1 || m1 == -1
        return QNVal(name(qv1), val(qv1) + fac * val(qv2), m1)
    end
    return QNVal(name(qv1), Base.mod(val(qv1) + fac * val(qv2), abs(m1)), m1)
end

Base.:(+)(qv1::QNVal, qv2::QNVal) = pm(qv1, qv2, +1)
Base.:(-)(qv1::QNVal, qv2::QNVal) = pm(qv1, qv2, -1)

const ZeroVal = QNVal()
