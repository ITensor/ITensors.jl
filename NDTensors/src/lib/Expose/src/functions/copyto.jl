function copyto!(R::Exposed, T::Exposed)
    copyto!(unexpose(R), unexpose(T))
    return unexpose(R)
end
