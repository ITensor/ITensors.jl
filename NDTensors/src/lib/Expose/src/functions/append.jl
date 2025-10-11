function Base.append!(Ecollection::Exposed, collections...)
    return append!(unexpose(Ecollection), collections...)
end
