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
Base.:-(qv::QNVal) =  QNVal(name(qv),-val(qv),modulus(qv))

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
  if m1 == 1
    return QNVal(name(qv1),val(qv1)+fac*val(qv2),1)
  end
  return QNVal(name(qv1),Base.mod(val(qv1)+fac*val(qv2),abs(m1)),m1)
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

store(qn::QN) = qn.store

Base.getindex(q::QN,n::Int) = getindex(store(q),n)

Base.length(qn::QN) = length(store(qn))

Base.lastindex(qn::QN) = length(qn)

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

#function valsMatch(x::QN,y::QN)
#  for xv in store(x)
#    @show xv
#    val(xv) == 0 && continue
#    found = false
#    for yv in store(y)
#      @show yv
#      name(yv)!=name(xv) && continue
#      val(yv)!=val(xv) && return false
#      found = true
#    end
#    found || return false
#  end
#  return true
#end

#function Base.:(==)(a::QN,b::QN)
#  return valsMatch(a,b) && valsMatch(b,a)
#end

function hasname(qn::QN,qv_find::QNVal)
  for qv in qn
    name(qv) == name(qv_find) && return true
  end
  return false
end

function Tensors.insertat(qn::QN,qv::QNVal,pos::Int)
  return QN(insertat(Tuple(qn),qv,pos))
end

function addqnval(qn::QN,qv_add::QNVal)
  isactive(qn[end]) && error("Cannot add QNVal, QN already contains maximum number of QNVals")
  for (pos,qv) in enumerate(qn)
    if qv_add < qv || !isactive(qv)
      return insertat(qn,qv_add,pos)
    end
  end
end

function fillqns(qn1::QN,qn2::QN)
  for qv2 in qn2
    if !hasname(qn1,qv2)
      qn1 = addqnval(qn1,zero(qv2))
    end
  end
  return qn1
end

function Base.:(==)(a::QN,b::QN)
  @show a
  @show b
  a_filled = fillqns(a,b)
  b_filled = fillqns(b,a)
  @show a_filled
  @show b_filled
  for (av,bv) in zip(a_filled,b_filled)
    av!=bv && return false
  end
  return true
end

function Base.:(<)(qa::QN,qb::QN)
  a = 1
  b = 1
  while a<=maxQNs && b<=maxQNs && (isactive(qa[a])||isactive(qb[b]))
    aval = val(qa[a])
    bval = val(qb[b])
    if !isactive(qa[a])
      if 0 == bval
        b += 1
        continue
      end
      return 0 < bval
    elseif !isactive(qb[b])
      if 0 == aval
        a += 1
        continue
      end
      return aval < 0
    else # both are active
      aname = name(qa[a])
      bname = name(qb[b])
      if aname < bname
        if aval == 0
          a += 1
          continue
        end
        return aval < 0
      elseif bname < aname
        if 0 == bval
          b += 1
          continue
        end
        return 0 < bval
      else  # aname == bname
        if aval == bval
          a += 1
          b += 1
          continue
        end
        return aval < bval
      end
    end
  end
  return false
end

function Base.show(io::IO,q::QN)
  print(io,"QN(")
  for n=1:maxQNs
    v = q[n]
    !isactive(v) && break
    n > 1 && print(io,",")
    print(io,"(\"$(name(v))\",$(val(v))")
    if modulus(v) != 1
      print(io,",$(modulus(v)))")
    else
      print(io,")")
    end
  end
  print(io,")")
end
