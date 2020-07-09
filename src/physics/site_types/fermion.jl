
function space(::SiteType"Fermion"; 
               conserve_qns=false,
               conserve_nf=conserve_qns,
               conserve_parity=conserve_qns)
  if conserve_nf
    zer = QN("Nf",0,-1) => 1
    one = QN("Nf",1,-1) => 1
    return [zer,one]
  elseif conserve_parity
    zer = QN("Pf",0,-2) => 1
    one = QN("Pf",1,-2) => 1
    return [zer,one]
  end
  return 2
end

state(::SiteType"Fermion",::StateName"Emp")  = 1
state(::SiteType"Fermion",::StateName"Occ")  = 2
state(st::SiteType"Fermion",::StateName"0") = state(st,StateName("Emp"))
state(st::SiteType"Fermion",::StateName"1") = state(st,StateName("Occ"))

function op!(Op::ITensor,
             ::OpName"N",
             ::SiteType"Fermion",
             s::Index)
  Op[s'=>2,s=>2] = 1.0
end

function op!(Op::ITensor,
             ::OpName"C",
             ::SiteType"Fermion",
             s::Index)
  Op[s'=>1,s=>2] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Cdag",
             ::SiteType"Fermion",
             s::Index)
  Op[s'=>2,s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"F",
             ::SiteType"Fermion",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>2,s=>2] = -1.0
end


has_fermion_string(::OpName"C",
                   ::SiteType"Fermion") = true

has_fermion_string(::OpName"Cdag",
                   ::SiteType"Fermion") = true

