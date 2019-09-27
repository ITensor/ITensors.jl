export QNVal,
       QN,
       name,
       val,
       modulus,
       isActive,
       isFermionic

const QNVal = Tuple{SmallString,Int,Int}

name(qv::QNVal) = qv[1]
val(qv::QNVal) = qv[2]
modulus(qv::QNVal) = qv[3]
isActive(qv::QNVal) = (modulus(qv) != 0)
isFermionic(qv::QNVal) = (modulus(qv) < 0)
Base.:<(qv1::QNVal,qv2::QNVal) = (name(qv1) < name(qv2))

function Base.:+(qv1::QNVal,qv2::QNVal) 
  if name(qv1) != name(qv2)
    error("Cannot add QNVals with different names \"$(name(qv1))\", \"$(name(qv2))\"")
  end
  if modulus(qv1) != modulus(qv2)
    error("QNVals with matching name \"$(name(qv1))\" cannot have different modulus values ")
  end
  m1 = modulus(qv1)
  if m1 == 1
    return (name(qv1),val(qv1)+val(qv2),1)
  end
  return (name(qv1),Base.mod(val(qv1)+val(qv2),abs(m1)),m1)
end

QNVal() = (SmallString(),0,0)
QNVal(t::Tuple{String,Int,Int}) = (SmallString(t[1]),t[2],t[3])
QNVal(t::Tuple{String,Int}) = (SmallString(t[1]),t[2],1)

const ZeroVal = (SmallString(),0,0)

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
    m[n] = QNVal(qv)
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

function Base.:+(a::QN,b::QN)
  isActive(b[1]) || return

  ma = MQNStorage(a.store)

  for nb=1:maxQNs
    bname = name(b[nb])
    for na=1:maxQNs
      aname = name(a[na])
      if !isActive(ma[na])
        ma[na] = bname
        break
      elseif name(ma[na]) == bname
        ma[na] += b[nb]
        break
      else bname < aname && (n==1 || bname > name(ma[na-1]))
        for j=maxQNs-1:-1:(na+1)
          ma[j] = ma[j-1]
        end
        ma[na] = b[nb]
      end
    end
  end

end

