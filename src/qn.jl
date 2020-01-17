export QNVal,
       QN,
       name,
       val,
       modulus,
       isactive,
       isfermionic

struct QNVal 
  name::SmallString
  val::Int
  modulus::Int

  function QNVal(name,v::Int,m::Int=1)
    am = abs(m)
    if am > 1
      new(SmallString(name),mod(v,am),m)
    end
    new(SmallString(name),v,m)
  end

end

QNVal(v::Int,m::Int=1) = QNVal("",v,m)
QNVal() = QNVal("",0,0)

name(qv::QNVal) = qv.name
val(qv::QNVal) = qv.val
modulus(qv::QNVal) = qv.modulus
isactive(qv::QNVal) = (modulus(qv) != 0)
isfermionic(qv::QNVal) = (modulus(qv) < 0)
Base.:<(qv1::QNVal,qv2::QNVal) = (name(qv1) < name(qv2))

function qn_mod(val::Int,modulus::Int)
  modulus = abs(modulus)
  (modulus == 0 || modulus == 1) && return val
  return mod(val,modulus)
end

function Base.:-(qv::QNVal)
  return QNVal(name(qv),qn_mod(-val(qv),modulus(qv)),modulus(qv))
end

Base.zero(qv::QNVal) = QNVal(name(qv),0,modulus(qv))

function Base.:*(dir::Arrow,qv::QNVal)
  return QNVal(name(qv),Int(dir)*val(qv),modulus(qv))
end

function pm(qv1::QNVal,qv2::QNVal,fac::Int) 
  if name(qv1) != name(qv2)
    error("Cannot add QNVals with different names \"$(name(qv1))\", \"$(name(qv2))\"")
  end
  if modulus(qv1) != modulus(qv2)
    error("QNVals with matching name \"$(name(qv1))\" cannot have different modulus values ")
  end
  m1 = modulus(qv1)
  return QNVal(name(qv1),qn_mod(val(qv1)+fac*val(qv2),m1),m1)
end

Base.:+(qv1::QNVal,qv2::QNVal) = pm(qv1,qv2,+1)
Base.:-(qv1::QNVal,qv2::QNVal) = pm(qv1,qv2,-1)

const ZeroVal = QNVal()

const maxQNs = 4
const QNStorage = SVector{maxQNs,QNVal}
const MQNStorage = MVector{maxQNs,QNVal}

struct QN
  store::QNStorage

  function QN()
    s = QNStorage(ntuple(_ ->ZeroVal,Val(maxQNs)))
    new(s)
  end

  QN(s::QNStorage) = new(s)
end

QN(mqn::MQNStorage) = QN(QNStorage(mqn))
QN(mqn::NTuple{N,QNVal}) where {N} = QN(QNStorage(mqn))

function QN(qvs...)
  m = MQNStorage(ntuple(_->ZeroVal,Val(maxQNs)))
  for (n,qv) in enumerate(qvs)
    m[n] = QNVal(qv...)
  end
  Nvals = length(qvs)
  sort!(@view m[1:Nvals];by=name,alg=InsertionSort)
  for n=1:(length(qvs)-1)
    if name(m[n])==name(m[n+1])
      error("Duplicate name \"$(name(m[n]))\" in QN")
    end
  end
  return QN(QNStorage(m))
end

QN(name,val::Int,modulus::Int=1) = QN((name,val,modulus))
QN(val::Int,modulus::Int=1) = QN(("",val,modulus))

Tensors.store(qn::QN) = qn.store

Base.getindex(q::QN,n::Int) = getindex(store(q),n)

Base.length(qn::QN) = length(store(qn))

Base.lastindex(qn::QN) = length(qn)

isactive(qn::QN) = isactive(qn[1])

function Base.iterate(qn::QN,state::Int=1)
  (state > length(qn)) && return nothing
  return (qn[state],state+1)
end

function val(q::QN,name_)
  sname = SmallString(name_)
  for n=1:maxQNs
    name(q[n]) == sname && return val(q[n])
  end
  return 0
end

function modulus(q::QN,name_)
  sname = SmallString(name_)
  for n=1:maxQNs
    name(q[n]) == sname && return modulus(q[n])
  end
  return 0
end

function combineqns(a::QN,b::QN,operation)
  !isactive(b[1]) && return a

  ma = MQNStorage(store(a))

  for nb=1:maxQNs
    !isactive(b[nb]) && break
    bname = name(b[nb])
    for na=1:maxQNs
      aname = name(a[na])
      if !isactive(ma[na])
        ma[na] = b[nb]
        break
      elseif name(ma[na]) == bname
        ma[na] = operation(ma[na],b[nb])
        break
      elseif (bname < aname) && (na==1 || bname > name(ma[na-1]))
        for j=maxQNs:-1:(na+1)
          ma[j] = ma[j-1]
        end
        ma[na] = b[nb]
        break
      end
    end
  end
  return QN(QNStorage(ma))
end

function Base.:*(dir::Arrow,qn::QN)
  mqn = MQNStorage(undef)
  for i in 1:length(mqn)
    mqn[i] = dir*qn[i]
  end
  return QN(mqn)
end

function Base.:+(a::QN,b::QN)
  return combineqns(a,b,+)
end

function Base.:-(a::QN,b::QN)
  return combineqns(a,b,-)
end

function Base.:-(qn::QN)
  mqn = MQNStorage(undef)
  for i in 1:length(mqn)
    mqn[i] = -qn[i]
  end
  return QN(mqn)
end

function hasname(qn::QN,qv_find::QNVal)
  for qv in qn
    name(qv) == name(qv_find) && return true
  end
  return false
end

# Does not perform checks on if QN is already full, drops
# the last QNVal
function Tensors.insertafter(qn::QN,qv::QNVal,pos::Int)
  return QN(insertafter(Tuple(qn),qv,pos)[1:length(qn)])
end

function addqnval(qn::QN,qv_add::QNVal)
  isactive(qn[end]) && error("Cannot add QNVal, QN already contains maximum number of QNVals")
  for (pos,qv) in enumerate(qn)
    if qv_add < qv || !isactive(qv)
      return insertafter(qn,qv_add,pos-1)
    end
  end
end

# Fills in the qns of qn1 that qn2 has but
# qn1 doesn't
function fillqns_from(qn1::QN,qn2::QN)
  # If qn1 has no non-trivial qns, fill
  # with qn2
  !isactive(qn1) && return qn2
  for qv2 in qn2
    if !hasname(qn1,qv2)
      qn1 = addqnval(qn1,zero(qv2))
    end
  end
  return qn1
end

# Make sure qn1 and qn2 have all of the same qns
function fillqns(qn1::QN,qn2::QN)
  qn1_filled = fillqns_from(qn1,qn2)
  qn2_filled = fillqns_from(qn2,qn1)
  return qn1_filled,qn2_filled
end

function isequal_assume_filled(qn1::QN,qn2::QN)
  for (qv1,qv2) in zip(qn1,qn2)
    modulus(qv1)!=modulus(qv2) && error("QNVals must have same modulus to compare")
    qv1!=qv2 && return false
  end
  return true
end

function Base.:(==)(qn1::QN,qn2::QN; assume_filled=false)
  if !assume_filled
    qn1,qn2 = fillqns(qn1,qn2)
  end
  return isequal_assume_filled(qn1,qn2)
end

function isless_assume_filled(qn1::QN,qn2::QN)
  for n in 1:length(qn1)
    val1 = val(qn1[n])
    val2 = val(qn2[n])
    val1 != val2 && return val1 < val2
  end
  return false
end

function Base.isless(qn1::QN,qn2::QN; assume_filled=false)
  return <(qn1,qn2;assume_filled=assume_filled)
end

function Base.:<(qn1::QN,qn2::QN; assume_filled=false)
  if !assume_filled
    qn1,qn2 = fillqns(qn1,qn2)
  end
  return isless_assume_filled(qn1,qn2)
end

function have_same_qns(qn1::QN,qn2::QN)
  for n in 1:length(qn1)
    name(qn1[n]) != name(qn2[n]) && return false
  end
  return true
end

function have_same_mods(qn1::QN,qn2::QN)
  for n in 1:length(qn1)
    modulus(qn1[n]) != modulus(qn2[n]) && return false
  end
  return true
end

function Base.show(io::IO,q::QN)
  print(io,"QN(")
  for n=1:maxQNs
    v = q[n]
    !isactive(v) && break
    n > 1 && print(io,",")
    if name(v)==SmallString("")
      print(io,"($(val(v))")
    else
      print(io,"(\"$(name(v))\",$(val(v))")
    end
    if modulus(v) != 1
      print(io,",$(modulus(v)))")
    else
      print(io,")")
    end
  end
  print(io,")")
end

