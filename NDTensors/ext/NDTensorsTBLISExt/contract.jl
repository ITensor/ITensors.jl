# Loading `TBLIS.jl` activates this `is_applicable` overload, which
# is what gates `with_tblis` scopes from picking `TBLIS()` for inputs
# the impl can't actually handle. Without the extension loaded, the
# baseline `is_applicable(::TBLIS, ::Type, ::Type) = false` in
# NDTensors core keeps `with_tblis` scopes inert.
function NDTensors.is_applicable(
        ::NDTensors.TBLIS,
        T1::Type{<:DenseTensor{<:LinearAlgebra.BlasReal}},
        T2::Type{<:DenseTensor{<:LinearAlgebra.BlasReal}}
    )
    return eltype(T1) === eltype(T2)
end

function NDTensors.contract!(
        ::NDTensors.TBLIS,
        R::DenseTensor{ElT},
        labelsR,
        T1::DenseTensor{ElT},
        labelsT1,
        T2::DenseTensor{ElT},
        labelsT2,
        α::Number = one(ElT),
        β::Number = zero(ElT)
    ) where {ElT <: LinearAlgebra.BlasReal}
    # TBLIS Tensors
    R_tblis = TBLIS.TTensor{ElT}(array(R), β)
    T1_tblis = TBLIS.TTensor{ElT}(array(T1), α)
    T2_tblis = TBLIS.TTensor{ElT}(array(T2))

    function label_to_char(label)
        # Start at 'a'
        char_start = Char(96)
        if label < 0
            # Start at 'z'
            char_start = Char(123)
        end
        return char_start + label
    end

    function labels_to_tblis(labels)
        if isempty(labels)
            return ""
        end
        str = prod(label_to_char.(labels))
        return str
    end

    labelsT1_tblis = labels_to_tblis(labelsT1)
    labelsT2_tblis = labels_to_tblis(labelsT2)
    labelsR_tblis = labels_to_tblis(labelsR)

    TBLIS.mul!(R_tblis, T1_tblis, T2_tblis, labelsT1_tblis, labelsT2_tblis, labelsR_tblis)

    return R
end
