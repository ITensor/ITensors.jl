function op!(Op::ITensor, ::OpName"Id", ::SiteType"Generic", s::Index)
  for n in 1:dim(s)
    Op[n, n] = 1.0
  end
end

op(::OpName"I", t::SiteType"Generic", s::Index) = op("Id", s)

function op!(Op::ITensor, ::OpName"F", st::SiteType"Generic", s::Index)
  return op!(Op, OpName("Id"), st, s)
end
