function copyto!(R::Exposed, T::Exposed)
  Base.copyto!(unexpose(R), unexpose(T))
  return unexpose(R)
end
