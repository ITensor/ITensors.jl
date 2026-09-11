function contract!(
        ::Val{:TBLIS},
        R::DenseTensor{ElT},
        labelsR,
        T1::DenseTensor{ElT},
        labelsT1,
        T2::DenseTensor{ElT},
        labelsT2,
        α::ElT,
        β::ElT
    ) where {ElT <: LinearAlgebra.BlasReal}
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

    # `R := β R + α T1 T2`
    TBLIS.tblis_tensor_mult(
        α, array(T1), labelsT1_tblis,
        array(T2), labelsT2_tblis,
        β, array(R), labelsR_tblis
    )

    return R
end
