struct QNVal
  name::SmallString
  val::Int
  modulus::Int

  function QNVal(name, v::Int, m::Int=1)
    am = abs(m)
    if am > 1
      return new(SmallString(name), mod(v, am), m)
    end
    return new(SmallString(name), v, m)
  end
end

QNVal(v::Int, m::Int=1) = QNVal("", v, m)
QNVal() = QNVal("", 0, 0)

name(qv::QNVal) = qv.name
val(qv::QNVal) = qv.val
modulus(qv::QNVal) = qv.modulus
isactive(qv::QNVal) = modulus(qv) != 0
(qv1::QNVal < qv2::QNVal) = (name(qv1) < name(qv2))

function qn_mod(val::Int, modulus::Int)
  amod = abs(modulus)
  amod <= 1 && return val
  return mod(val, amod)
end

function -(qv::QNVal)
  return QNVal(name(qv), qn_mod(-val(qv), modulus(qv)), modulus(qv))
end

zero(qv::QNVal) = QNVal(name(qv), 0, modulus(qv))

(dir::Arrow * qv::QNVal) = QNVal(name(qv), Int(dir) * val(qv), modulus(qv))

(qv::QNVal * dir::Arrow) = (dir * qv)

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

(qv1::QNVal + qv2::QNVal) = pm(qv1, qv2, +1)
(qv1::QNVal - qv2::QNVal) = pm(qv1, qv2, -1)

const ZeroVal = QNVal()

const maxQNs = 4
const QNStorage = SVector{maxQNs,QNVal}
const MQNStorage = MVector{maxQNs,QNVal}

"""
A QN object stores a collection of up to four
named values such as ("Sz",1) or ("N",0).
These values can include a third integer "m"
which makes them obey addition modulo m, for
example ("P",1,2) for a value obeying addition mod 2.
(The default is regular integer addition).

Adding or subtracting pairs of QN objects performs
addition and subtraction element-wise on each of
the named values. If a name is missing from the
collection, its value is treated as zero.
"""
struct QN
  data::QNStorage

  function QN()
    s = QNStorage(ntuple(_ -> ZeroVal, Val(maxQNs)))
    return new(s)
  end

  QN(s::QNStorage) = new(s)
end

QN(mqn::MQNStorage) = QN(QNStorage(mqn))
QN(mqn::NTuple{N,QNVal}) where {N} = QN(QNStorage(mqn))

function hash(obj::QN, h::UInt)
  # TODO: use an MVector or SVector
  # for performance here; put non-zero QNVals
  # to front and slice when passing to hash
  nzqv = QNVal[]
  for qv in obj.data
    if val(qv) != 0
      push!(nzqv, qv)
    end
  end
  return hash(nzqv, h)
end

"""
    QN(qvs...)

Construct a QN from a set of up to four
named value tuples.

Examples

```julia
q = QN(("Sz",1))
q = QN(("N",1),("Sz",-1))
q = QN(("P",0,2),("Sz",0)).
```
"""
function QN(qvs...)
  m = MQNStorage(ntuple(_ -> ZeroVal, Val(maxQNs)))
  for (n, qv) in enumerate(qvs)
    m[n] = QNVal(qv...)
  end
  Nvals = length(qvs)
  sort!(@view m[1:Nvals]; by=name, alg=InsertionSort)
  for n in 1:(length(qvs) - 1)
    if name(m[n]) == name(m[n + 1])
      error("Duplicate name \"$(name(m[n]))\" in QN")
    end
  end
  return QN(QNStorage(m))
end

"""
    QN(name,val::Int,modulus::Int=1)

Construct a QN with a single named value
by providing the name, value, and optional
modulus.
"""
QN(name, val::Int, modulus::Int=1) = QN((name, val, modulus))

"""
    QN(val::Int,modulus::Int=1)

Construct a QN with a single unnamed value
(equivalent to the name being the empty string)
with optional modulus.
"""
QN(val::Int, modulus::Int=1) = QN(("", val, modulus))

data(qn::QN) = qn.data

getindex(q::QN, n::Int) = getindex(data(q), n)

length(qn::QN) = length(data(qn))

lastindex(qn::QN) = length(qn)

isactive(qn::QN) = isactive(qn[1])

function nactive(q::QN)
  for n in 1:maxQNs
    !isactive(q[n]) && (return n - 1)
  end
  return maxQNs
end

function iterate(qn::QN, state::Int=1)
  (state > length(qn)) && return nothing
  return (qn[state], state + 1)
end

keys(qn::QN) = keys(data(qn))

"""
    val(q::QN,name)

Get the value within the QN q
corresponding to the string `name`
"""
function val(q::QN, name_)
  sname = SmallString(name_)
  for n in 1:maxQNs
    name(q[n]) == sname && return val(q[n])
  end
  return 0
end

"""
    modulus(q::QN,name)

Get the modulus within the QN q
corresponding to the string `name`
"""
function modulus(q::QN, name_)
  sname = SmallString(name_)
  for n in 1:maxQNs
    name(q[n]) == sname && return modulus(q[n])
  end
  return 0
end

"""
    zero(q::QN)

Returns a QN object containing
the same names as q, but with
all values set to zero.
"""
function zero(qn::QN)
  mqn = MQNStorage(undef)
  for i in 1:length(mqn)
    mqn[i] = zero(qn[i])
  end
  return QN(mqn)
end

function (dir::Arrow * qn::QN)
  mqn = MQNStorage(undef)
  for i in 1:length(mqn)
    mqn[i] = dir * qn[i]
  end
  return QN(mqn)
end

(qn::QN * dir::Arrow) = (dir * qn)

function -(qn::QN)
  mqn = MQNStorage(undef)
  for i in 1:length(mqn)
    mqn[i] = -qn[i]
  end
  return QN(mqn)
end

function (a::QN + b::QN)
  !isactive(b[1]) && return a

  ma = MQNStorage(data(a))
  for nb in 1:maxQNs
    !isactive(b[nb]) && break
    bname = name(b[nb])
    for na in 1:maxQNs
      aname = name(a[na])
      if !isactive(ma[na])
        ma[na] = b[nb]
        break
      elseif name(ma[na]) == bname
        ma[na] = ma[na] + b[nb]
        break
      elseif (bname < aname) && (na == 1 || bname > name(ma[na - 1]))
        for j in maxQNs:-1:(na + 1)
          ma[j] = ma[j - 1]
        end
        ma[na] = b[nb]
        break
      end
    end
  end
  return QN(QNStorage(ma))
end

(a::QN - b::QN) = (a + (-b))

function hasname(qn::QN, qv_find::QNVal)
  for qv in qn
    name(qv) == name(qv_find) && return true
  end
  return false
end

# Does not perform checks on if QN is already full, drops
# the last QNVal
# Rename insert?
function NDTensors.insertafter(qn::QN, qv::QNVal, pos::Int)
  return QN(NDTensors.insertafter(Tuple(qn), qv, pos)[1:length(qn)])
end

function addqnval(qn::QN, qv_add::QNVal)
  isactive(qn[end]) &&
    error("Cannot add QNVal, QN already contains maximum number of QNVals")
  for (pos, qv) in enumerate(qn)
    if qv_add < qv || !isactive(qv)
      return NDTensors.insertafter(qn, qv_add, pos - 1)
    end
  end
end

# Fills in the qns of qn1 that qn2 has but
# qn1 doesn't
function fillqns_from(qn1::QN, qn2::QN)
  # If qn1 has no non-trivial qns, fill
  # with qn2
  !isactive(qn1) && return zero(qn2)
  for qv2 in qn2
    if !hasname(qn1, qv2)
      qn1 = addqnval(qn1, zero(qv2))
    end
  end
  return qn1
end

# Make sure qn1 and qn2 have all of the same qns
function fillqns(qn1::QN, qn2::QN)
  qn1_filled = fillqns_from(qn1, qn2)
  qn2_filled = fillqns_from(qn2, qn1)
  return qn1_filled, qn2_filled
end

function isequal_assume_filled(qn1::QN, qn2::QN)
  for (qv1, qv2) in zip(qn1, qn2)
    modulus(qv1) != modulus(qv2) && error("QNVals must have same modulus to compare")
    qv1 != qv2 && return false
  end
  return true
end

function ==(qn1::QN, qn2::QN; assume_filled=false)
  if !assume_filled
    qn1, qn2 = fillqns(qn1, qn2)
  end
  return isequal_assume_filled(qn1, qn2)
end

function isless_assume_filled(qn1::QN, qn2::QN)
  for n in 1:length(qn1)
    val1 = val(qn1[n])
    val2 = val(qn2[n])
    val1 != val2 && return val1 < val2
  end
  return false
end

function isless(qn1::QN, qn2::QN; assume_filled=false)
  return <(qn1, qn2; assume_filled=assume_filled)
end

function <(qn1::QN, qn2::QN; assume_filled=false)
  if !assume_filled
    qn1, qn2 = fillqns(qn1, qn2)
  end
  return isless_assume_filled(qn1, qn2)
end

function have_same_qns(qn1::QN, qn2::QN)
  for n in 1:length(qn1)
    name(qn1[n]) != name(qn2[n]) && return false
  end
  return true
end

function have_same_mods(qn1::QN, qn2::QN)
  for n in 1:length(qn1)
    modulus(qn1[n]) != modulus(qn2[n]) && return false
  end
  return true
end

function removeqn(qn::QN, qn_name::String)
  ss_qn_name = SmallString(qn_name)

  # Find the location of the QNVal to remove
  n_qn = nothing
  for n in 1:length(qn)
    qnval = qn[n]
    if name(qnval) == ss_qn_name
      n_qn = n
    end
  end
  if isnothing(n_qn)
    return qn
  end

  qn_data = data(qn)
  for j in n_qn:(length(qn) - 1)
    qn_data = setindex(qn_data, qn_data[j + 1], j)
  end
  qn_data = setindex(qn_data, QNVal(), length(qn))
  return QN(qn_data)
end

function show(io::IO, q::QN)
  print(io, "QN(")
  Na = nactive(q)
  for n in 1:Na
    v = q[n]
    n > 1 && print(io, ",")
    Na > 1 && print(io, "(")
    if name(v) != SmallString("")
      print(io, "\"$(name(v))\",")
    end
    print(io, "$(val(v))")
    if modulus(v) != 1
      print(io, ",$(modulus(v))")
    end
    Na > 1 && print(io, ")")
  end
  return print(io, ")")
end

function write(parent::Union{HDF5.File,HDF5.Group}, gname::AbstractString, q::QN)
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

function read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{QN})
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "QN"
    error("HDF5 group or file does not contain QN data")
  end
  names = read(g, "names")
  vals = read(g, "vals")
  mods = read(g, "mods")
  mqn = ntuple(n -> QNVal(names[n], vals[n], mods[n]), maxQNs)
  return QN(mqn)
end
