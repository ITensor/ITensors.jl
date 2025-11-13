function mul!(CM::Exposed, AM::Exposed, BM::Exposed, α, β)
    mul!(unexpose(CM), unexpose(AM), unexpose(BM), α, β)
    return unexpose(CM)
end
