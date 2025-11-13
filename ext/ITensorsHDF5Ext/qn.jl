using HDF5: HDF5, attributes, create_group, open_group, read, write
using ITensors: maxQNs, modulus, name, QN, QNVal, val

function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, gname::AbstractString, q::QN)
    g = create_group(parent, gname)
    attributes(g)["type"] = "QN"
    attributes(g)["version"] = 1
    names = [String(name(q[n])) for n in 1:maxQNs]
    vals = [val(q[n]) for n in 1:maxQNs]
    mods = [modulus(q[n]) for n in 1:maxQNs]
    write(g, "names", names)
    write(g, "vals", vals)
    return write(g, "mods", mods)
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{QN})
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "QN"
    error("HDF5 group or file does not contain QN data")
  end
  names = read(g, "names")
  vals = read(g, "vals")
  mods = read(g, "mods")
  nemptyQN = maxQNs - length(names)
  if (nemptyQN > 0)
    append!(names, fill("", nemptyQN))
    append!(vals, fill(0, nemptyQN))
    append!(mods, fill(0, nemptyQN))
  end
  mqn = ntuple(n -> QNVal(names[n], vals[n], mods[n]), maxQNs)
  return QN(mqn)
end
