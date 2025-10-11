function mul!(CM::Exposed{<:Array}, AM::Exposed{<:Array}, BM::Exposed{<:Array}, α, β)
    @strided mul!(unexpose(CM), unexpose(AM), unexpose(BM), α, β)
    return unexpose(CM)
end
