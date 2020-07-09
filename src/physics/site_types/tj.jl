
function space(::SiteType"tJ";
               conserve_qns=false,
               conserve_sz=conserve_qns,
               conserve_nf=conserve_qns,
               conserve_parity=conserve_qns)
  if conserve_sz && conserve_nf
    em = QN(("Nf",0,-1),("Sz", 0)) => 1
    up = QN(("Nf",1,-1),("Sz",+1)) => 1
    dn = QN(("Nf",1,-1),("Sz",-1)) => 1
    return [em,up,dn]
  elseif conserve_nf
    zer = QN("Nf",0,-1) => 1
    one = QN("Nf",1,-1) => 2
    return [zer,one]
  elseif conserve_sz
    em = QN(("Sz", 0),("Pf",0,-2)) => 1
    up = QN(("Sz",+1),("Pf",1,-2)) => 1
    dn = QN(("Sz",-1),("Pf",1,-2)) => 1
    return [em,up,dn]
  elseif conserve_parity
    zer = QN("Pf",0,-2) => 1
    one = QN("Pf",1,-2) => 2
    return [zer,one]
  end
  return 3
end

state(::SiteType"tJ",::StateName"Emp")  = 1
state(::SiteType"tJ",::StateName"Up")   = 2
state(::SiteType"tJ",::StateName"Dn")   = 3
state(st::SiteType"tJ",::StateName"0")    = state(st,StateName("Emp"))
state(st::SiteType"tJ",::StateName"↑")    = state(st,StateName("Up"))
state(st::SiteType"tJ",::StateName"↓")    = state(st,StateName("Dn"))

function op!(Op::ITensor,
             ::OpName"Nup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>2] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Ndn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>3,s=>3] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Ntot",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>2] = 1.0
  Op[s'=>3,s=>3] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Cup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>2] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Cdagup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Cdn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>3] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Cdagdn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>3,s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Aup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>2] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Adagup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Adn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>3] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Adagdn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>3,s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"F",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>2,s=>2] = -1.0
  Op[s'=>3,s=>3] = -1.0
end

function op!(Op::ITensor,
             ::OpName"Fup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>2,s=>2] = -1.0
  Op[s'=>3,s=>3] = +1.0
end

function op!(Op::ITensor,
             ::OpName"Fdn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>2,s=>2] = +1.0
  Op[s'=>3,s=>3] = -1.0
end

function op!(Op::ITensor,
             ::OpName"Sz",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>2] = +0.5
  Op[s'=>3,s=>3] = -0.5
end

op!(Op::ITensor,
    ::OpName"Sᶻ",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("Sz"),st,s)

function op!(Op::ITensor,
             ::OpName"Sx",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>3] = 0.5
  Op[s'=>3,s=>2] = 0.5
end

op!(Op::ITensor,
    ::OpName"Sˣ",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("Sx"),st,s)

function op!(Op::ITensor,
             ::OpName"S+",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>3] = 1.0
end

op!(Op::ITensor,
    ::OpName"S⁺",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S+"),st,s)
op!(Op::ITensor,
    ::OpName"Sp",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S+"),st,s)
op!(Op::ITensor,
    ::OpName"Splus",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S+"),st,s)

function op!(Op::ITensor,
             ::OpName"S-",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>3,s=>2] = 1.0
end

op!(Op::ITensor,
    ::OpName"S⁻",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S-"),st,s)
op!(Op::ITensor,
    ::OpName"Sm",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S-"),st,s)
op!(Op::ITensor,
    ::OpName"Sminus",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S-"),st,s)

has_fermion_string(::OpName"Cup", ::SiteType"tJ") = true
has_fermion_string(::OpName"Cdagup", ::SiteType"tJ") = true
has_fermion_string(::OpName"Cdn", ::SiteType"tJ") = true
has_fermion_string(::OpName"Cdagdn", ::SiteType"tJ") = true
