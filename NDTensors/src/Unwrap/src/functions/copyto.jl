function copyto!(R::Exposed, T::Exposed)
  Base.copyto!(R.object, T.object)
  return R
end
