using HDF5: HDF5, attributes, create_group, open_group, read, read_attribute, write
using ITensors: QN, QNVal, maxQNs, modulus, name, val

function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, gname::AbstractString, q::QN)
    return closeafter(create_group(parent, gname)) do g
        attributes(g)["type"] = "QN"
        attributes(g)["version"] = 1
        names = [String(name(q[n])) for n in 1:maxQNs]
        vals = [val(q[n]) for n in 1:maxQNs]
        mods = [modulus(q[n]) for n in 1:maxQNs]
        write(g, "names", names)
        write(g, "vals", vals)
        return write(g, "mods", mods)
    end
end

function HDF5.read(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{QN})
    return closeafter(open_group(parent, name)) do g
        if read_attribute(g, "type") != "QN"
            error("HDF5 group or file does not contain QN data")
        end
        names = read(g, "names")
        vals = read(g, "vals")
        mods = read(g, "mods")
        mqn = ntuple(n -> QNVal(names[n], vals[n], mods[n]), maxQNs)
        return QN(mqn)
    end
end
