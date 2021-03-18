
"""
    space(::SiteType"S=1/2";
          conserve_qns = false,
          conserve_sz = conserve_qns,
          conserve_szparity = false,
          qnname_sz = "Sz",
          qnname_szparity = "SzParity")

Create the Hilbert space for a site of type "S=1/2".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function space(::SiteType"S=1/2";
               conserve_qns = false,
               conserve_sz = conserve_qns,
               conserve_szparity = false,
               qnname_sz = "Sz",
               qnname_szparity = "SzParity")
  if conserve_sz && conserve_szparity
    return [QN((qnname_sz, +1), (qnname_szparity, 1, 2)) => 1,
            QN((qnname_sz, -1), (qnname_szparity, 0, 2)) => 1]
  elseif conserve_sz
    return [QN(qnname_sz, +1) => 1,
            QN(qnname_sz, -1) => 1]
  elseif conserve_szparity
    return [QN(qnname_szparity, 1, 2) => 1,
            QN(qnname_szparity, 0, 2) => 1]
  end
  return 2
end

state(::SiteType"S=1/2", ::StateName"Up") = 1
state(::SiteType"S=1/2", ::StateName"Dn") = 2

state(st::SiteType"S=1/2", ::StateName"↑") =
  state(st, StateName("Up"))
state(st::SiteType"S=1/2", ::StateName"↓") =
  state(st, StateName("Dn"))


op(::OpName"Z",::SiteType"S=1/2") =
  [1  0
   0 -1]


op(::OpName"Sz",::SiteType"S=1/2") =
  [0.5  0.0
   0.0 -0.5]

op(::OpName"Sᶻ",t::SiteType"S=1/2") = 
  op(OpName("Sz"),t)


op(::OpName"S+",::SiteType"S=1/2") =
  [0  1
   0  0]

op(::OpName"S⁺",t::SiteType"S=1/2") = 
  op(OpName("S+"),t)

op(::OpName"Splus",t::SiteType"S=1/2") = 
  op(OpName("S+"),t)


op(::OpName"S-",::SiteType"S=1/2") =
  [0  0
   1  0]

op(::OpName"S⁻",t::SiteType"S=1/2") = 
  op(OpName("S-"),t)

op(::OpName"Sminus",t::SiteType"S=1/2") = 
  op(OpName("S-"),t)


op(::OpName"X",::SiteType"S=1/2") =
  [0  1
   1  0]


op(::OpName"Sx",::SiteType"S=1/2") =
  [0.0  0.5
   0.5  0.0]

op(::OpName"Sˣ",t::SiteType"S=1/2") = 
  op(OpName("Sx"),t)

op(::OpName"iY",::SiteType"S=1/2") =
  [ 0 1
   -1 0]

op(::OpName"iSy",::SiteType"S=1/2") =
  [ 0.0  0.5
   -0.5  0.0]

op(::OpName"iSʸ",t::SiteType"S=1/2") = 
  op(OpName("iSy"),t)


op(::OpName"Y",::SiteType"S=1/2") =
  [0.0   -1.0im
   1.0im  0.0  ]


op(::OpName"Sy",::SiteType"S=1/2") =
  [0.0   -0.5im
   0.5im  0.0  ]

op(::OpName"Sʸ",t::SiteType"S=1/2") = 
  op(OpName("Sy"),t)


op(::OpName"S2",::SiteType"S=1/2") =
  [0.75  0.0
   0.0   0.75]

op(::OpName"S²",t::SiteType"S=1/2") = 
  op(OpName("S2"),t)


op(::OpName"ProjUp",::SiteType"S=1/2") =
  [1  0
   0  0]

op(::OpName"projUp",t::SiteType"S=1/2") = 
  op(OpName("ProjUp"),t)


op(::OpName"ProjDn",::SiteType"S=1/2") =
  [0  0
   0  1]

op(::OpName"projDn",t::SiteType"S=1/2") = 
  op(OpName("ProjDn"),t)


# Support the tag "SpinHalf" as equivalent to "S=1/2"

space(::SiteType"SpinHalf"; kwargs...) =
  space(SiteType("S=1/2"); kwargs...)

state(::SiteType"SpinHalf", n::StateName) =
  state(SiteType("S=1/2"), n)

op(o::OpName, ::SiteType"SpinHalf";kwargs...) =
  op(o, SiteType("S=1/2");kwargs...)

# Support the tag "S=½" as equivalent to "S=1/2"

space(::SiteType"S=½"; kwargs...) =
  space(SiteType("S=1/2"); kwargs...)

state(::SiteType"S=½", n::StateName) =
  state(SiteType("S=1/2"), n)

op(o::OpName, ::SiteType"S=½";kwargs...) =
  op(o, SiteType("S=1/2");kwargs...)

