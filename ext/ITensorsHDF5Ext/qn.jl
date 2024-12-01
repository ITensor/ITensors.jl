using HDF5: HDF5, attributes, create_group, open_group, read, write
using ITensors: maxQNs, modulus, name, QN, QNVal, val

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, gname::AbstractString, q::QN)
  g = create_group(parent, gname)
  attributes(g)["type"] = "QN"
  attributes(g)["version"] = 1
  names = [String(name(q[n])) for n in 1:max_qn]
  vals = [val(q[n]) for n in 1:max_qn]
  mods = [modulus(q[n]) for n in 1:max_qn]
  write(g, "names", names)
  write(g, "vals", vals)
  return write(g, "mods", mods)
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{QN} ; max_qn = maxQNs)
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "QN"
    error("HDF5 group or file does not contain QN data")
  end
  names = read(g, "names")
  vals = read(g, "vals")
  mods = read(g, "mods")
  mqn = ntuple(n -> QNVal(names[n], vals[n], mods[n]), max_qn)
  return QN(mqn)
end
