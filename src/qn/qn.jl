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

const ZeroVal = QNVal("", 0, 0)

"""
A QN object stores a collection of up to N
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
struct QN{T}
  data::T

  QN() = new{SVector{0,QNVal}}(SVector{0,QNVal}())
  QN(s::SVector{N,QNVal}) where {N} = new{SVector{N,QNVal}}(s)
  QN(m::MVector{N,QNVal}) where {N} = new{SVector{N,QNVal}}(SVector(m))
  QN(mqn::NTuple{N,QNVal}) where {N} = new{SVector{N,QNVal}}(SVector(mqn))
  QN(mqn::AbstractArray{<:QNVal}) = new{SVector{length(mqn),QNVal}}(SVector(mqn))
  QN(d::Dict{TKey,QNVal}) where {TKey} = new{Dict{TKey,QNVal}}(d)
end

function hash(obj::QN, h::UInt)
  # TODO: use an MVector or SVector
  # for performance here; put non-zero QNVals
  # to front and slice when passing to hash
  nzqv = QNVal[]
  for qv in data(obj)
    if val(qv) != 0
      push!(nzqv, qv)
    end
  end
  return hash(nzqv, h)
end

"""
    QN(qvs...)

Construct a QN from a set of N
named value tuples.

Examples

```julia
q = QN(("Sz",1))
q = QN(("N",1),("Sz",-1))
q = QN(("P",0,2),("Sz",0)).
```
"""
function QN(qvs...)
  m = MVector(ntuple(_ -> ZeroVal, length(qvs)))
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
  return QN(m)
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

data(qn::QN{<:AbstractArray}) = qn.data
function data(qn::QN{<:AbstractDict})
  return [p[2] for p in sort(collect(qn.data); by=p -> p[1])]
end

getindex(q::QN, n::Int) = getindex(data(q), n)

length(qn::QN) = length(data(qn))

lastindex(qn::QN) = length(qn)

isactive(qn::QN) = length(qn) > 0 && isactive(qn[1])

function nactive(q::QN)
  d = data(q)
  for n in 1:length(d)
    !isactive(d[n]) && (return n - 1)
  end
  return length(d)
end

function iterate(qn::QN, state::Int=1)
  d = data(qn)
  (state > length(d)) && return nothing
  return (d[state], state + 1)
end

keys(qn::QN) = keys(qn.data)

"""
    get_qn_val(q::QN, name)

Get the QNVal within the QN q
corresponding to the string `name`
"""
function get_qn_val(q::QN{<:AbstractArray{<:QNVal}}, name_)
  sname = SmallString(name_)
  for n in 1:length(q.data)
    name(q[n]) == sname && return q[n]
  end
  return nothing
end
function get_qn_val(q::QN{<:AbstractDict{TKey,<:QNVal}}, name_) where {TKey}
  sname = TKey(name_)
  if sname in keys(q)
    return q[sname]
  end
  return nothing
end

"""
    val(q::QN,name)

Get the value within the QN q
corresponding to the string `name`
"""
function val(q::QN, name_)
  qnval = get_qn_val(q, name_)
  return qnval === nothing ? 0 : val(qnval)
end

"""
    modulus(q::QN,name)

Get the modulus within the QN q
corresponding to the string `name`
"""
function modulus(q::QN, name_)
  qnval = get_qn_val(q, name_)
  return qnval === nothing ? 0 : modulus(qnval)
end

"""
    zero(q::QN)

Returns a QN object containing
the same names as q, but with
all values set to zero.
"""
function zero(qn::QN{<:AbstractArray})
  mqn = MVector{length(qn.data),QNVal}(undef)
  for i in 1:length(mqn)
    mqn[i] = zero(qn[i])
  end
  return QN(mqn)
end
function zero(qn::QN{<:AbstractDict})
  return QN(Dict(k => zero(v) for (k, v) in collect(qn.data)))
end

function (dir::Arrow * qn::QN{<:AbstractArray})
  mqn = MVector{length(qn.data),QNVal}(undef)
  for i in 1:length(mqn)
    mqn[i] = dir * qn[i]
  end
  return QN(mqn)
end
function (dir::Arrow * qn::QN{<:AbstractDict})
  return QN(k => dir * v for (k, v) in collect(qn))
end

(qn::QN * dir::Arrow) = (dir * qn)

function -(qn::QN{<:AbstractArray})
  mqn = MVector{length(qn.data),QNVal}(undef)
  for i in 1:length(mqn)
    mqn[i] = -qn[i]
  end
  return QN(mqn)
end
function -(qn::QN{<:AbstractDict})
  return QN(k => -v for (k, v) in collect(qn))
end

function (a::QN{<:AbstractArray} + b::QN{<:AbstractArray})
  # TODO can't get this to work without mapping to dictionaries

  # #length of keyset needed to determine length of merged vector
  # keyset = Set{SmallString}(name(x) for x in a.data)
  # for x in b.data
  #   isactive(x) && push!(keyset, name(x))
  # end

  # #create merged vector backfilled with a
  # ma = MVector{length(keyset),QNVal}(
  #   ntuple(i -> i < length(a) ? a[i] : ZeroVal, length(keyset))
  # )
  # for nb in 1:length(b)
  #   !isactive(b[nb]) && break
  #   bname = name(b[nb])
  #   for na in 1:length(a)
  #     aname = name(a[na])
  #     if !isactive(ma[na])
  #       ma[na] = b[nb]
  #       break
  #     elseif name(ma[na]) == bname
  #       ma[na] = ma[na] + b[nb]
  #       break
  #     elseif (bname < aname) && (na == 1 || bname > name(ma[na - 1]))
  #       for j in length(ma):-1:(na + 1)
  #         ma[j] = ma[j - 1]
  #       end
  #       ma[na] = b[nb]
  #       break
  #     end
  #   end
  # end
  # return QN(ma)

  #temporary hack -- move to dictionaries, add, move back to SVector
  qn = QN(Dict(name(v) => v for v in a.data)) + QN(Dict(name(v) => v for v in b.data))
  dqn = data(qn)
  return QN(SVector{length(dqn),QNVal}(dqn))
end
function (a::QN{<:AbstractDict} + b::QN{<:AbstractDict})
  !isactive(b) && return a
  !isactive(a) && return b

  #merge a and b dictionaries based on name, isactive
  ma_dict = Dict(k => v for (k, v) in collect(a.data))
  for kb in keys(b.data)
    !isactive(b.data[kb]) && break
    bname = name(b.data[kb])
    if !(bname in keys(ma_dict)) || (!isactive(ma_dict[bname]))
      ma_dict[bname] = b.data[kb]
    else
      ma_dict[bname] = ma_dict[bname] + b.data[bname]
    end
  end
  return QN(ma_dict)
end
function (a::QN{<:AbstractArray} + b::QN{<:AbstractDict})
  return QN(Dict(name(v) => v for v in data(a))) + b
end
function (a::QN{<:AbstractDict} + b::QN{<:AbstractArray})
  return a + QN(Dict(name(v) => v for v in data(b)))
end

(a::QN - b::QN) = (a + (-b))

function hasname(qn::QN, qv_find::QNVal)
  return get_qn_val(qn, name(qv_find)) !== nothing
end

# Does not perform checks on if QN is already full, drops
# the last QNVal
# Rename insert?
function NDTensors.insertafter(qn::QN{<:AbstractArray}, qv::QNVal, pos::Int)
  return QN(NDTensors.insertafter(Tuple(qn), qv, pos)[1:length(qn)])
end

function addqnval(qn::QN{<:AbstractArray}, qv_add::QNVal)::QN
  for (pos, qv) in enumerate(qn)
    if qv_add < qv || !isactive(qv)
      if pos == 1
        return QN(MVector{length(qn) + 1,QNVal}(qv_add, qn.data...))
      else
        return QN(
          MVector{length(qn) + 1,QNVal}(
            qn.data[1:(pos - 1)]..., qv_add, qn.data[pos:end]...
          ),
        )
      end
    end
  end
  return QN(MVector{length(qn) + 1,QNVal}(qn.data..., qv_add))
end
function addqnval(qn::QN{<:AbstractDict}, qv_add::QNVal)::QN
  d = Dict(qn.data)
  d[name(qv_add)] = qv_add
  return QN(d)
end

# Fills in the qns of qn1 that qn2 has but
# qn1 doesn't
function fillqns_from(qn1::QN, qn2::QN)::QN
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
function fillqns(qn1::QN, qn2::QN)::Tuple{QN,QN}
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

function removeqn(qn::QN{<:AbstractArray}, qn_name::String)
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
function removeqn(qn::QN{<:AbstractDict}, qn_name::String)
  ss_qn_name = SmallString(qn_name)
  return QN(k => v for (k, v) in collect(qn.data) if k != ss_qn_name)
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
  names = [String(name(qq)) for qq in q.data]
  vals = [val(qq) for qq in q.data]
  mods = [modulus(qq) for qq in q.data]
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
  mqn = ntuple(n -> QNVal(names[n], vals[n], mods[n]), length(names))
  return QN(mqn)
end
