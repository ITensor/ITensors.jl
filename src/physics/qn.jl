export QNVal,
       QN,
       name,
       val,
       modulus,
       isActive,
       isFermionic

struct QNVal 
  name::SmallString
  val::Int
  modulus::Int

  QNVal() = new(SmallString(),0,0)

  function QNVal(name::String,v::Int,m::Int=1)
    am = abs(m)
    if am > 1
      new(SmallString(name),mod(v,am),m)
    end
    new(SmallString(name),v,m)
  end

end

name(qv::QNVal) = qv.name
val(qv::QNVal) = qv.val
modulus(qv::QNVal) = qv.modulus
isActive(qv::QNVal) = (modulus(qv) != 0)
isFermionic(qv::QNVal) = (modulus(qv) < 0)
Base.:<(qv1::QNVal,qv2::QNVal) = (name(qv1) < name(qv2))
Base.:-(qv::QNVal) =  QNVal(name(qv),-val(qv),modulus(qv))

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

function QN(qvs...)
  m = MQNStorage(ntuple(_->ZeroVal,Val(maxQNs)))
  for (n,qv) in enumerate(qvs)
    m[n] = QNVal(qv...)
  end
  sort!(m;by=name,rev=true)
  for n=1:(length(qvs)-1)
    if name(m[n])==name(m[n+1])
      error("Duplicate name \"$(name(m[n]))\" in QN")
    end
  end
  return QN(QNStorage(m))
end

Base.getindex(q::QN,n::Int) = getindex(q.store,n)

function val(q::QN,name_)
  sname = SmallString(name_)
  name(q[1]) == sname && return val(q[1])
  name(q[2]) == sname && return val(q[2])
  name(q[3]) == sname && return val(q[3])
  name(q[4]) == sname && return val(q[4])
  error("QN does not have sector with name \"$name\"")
  return 0
end

function modulus(q::QN,name_)
  sname = SmallString(name_)
  name(q[1]) == sname && return modulus(q[1])
  name(q[2]) == sname && return modulus(q[2])
  name(q[3]) == sname && return modulus(q[3])
  name(q[4]) == sname && return modulus(q[4])
  error("QN does not have sector with name \"$name\"")
  return 0
end

function combineQNs(a::QN,b::QN,operation)
  !isActive(b[1]) && return a

  ma = MQNStorage(a.store)

  for nb=1:maxQNs
    !isActive(b[nb]) && break
    bname = name(b[nb])
    for na=1:maxQNs
      aname = name(a[na])
      if !isActive(ma[na])
        #println("Case 1")
        ma[na] = b[nb]
        break
      elseif name(ma[na]) == bname
        #println("Case 2")
        ma[na] = operation(ma[na],b[nb])
        break
      elseif (bname < aname) && (na==1 || bname > name(ma[na-1]))
        #println("Case 3")
        for j=maxQNs:-1:(na+1)
          #println("  Moving \"$(name(ma[j-1]))\" from $(j-1)->$j")
          ma[j] = ma[j-1]
        end
        ma[na] = b[nb]
        break
      end
    end
  end
  return QN(QNStorage(ma))
end

function Base.:+(a::QN,b::QN)
  return combineQNs(a,b,+)
end

function Base.:-(a::QN,b::QN)
  return combineQNs(a,b,-)
end

function Base.show(io::IO,q::QN)
  print(io,"QN(")
  for n=1:maxQNs
    v = q[n]
    !isActive(v) && break
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
