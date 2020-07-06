
function op(::OpName"Id",
            ::SiteType,
            s::Index)
  Op = emptyITensor(s', dag(s))
  for n=1:dim(s)
    Op[n,n] = 1.0
  end
  return Op
end

