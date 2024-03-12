using StridedViews: StridedView

@eval position(type::Type{<:StridedView}, ::typeof(parenttype)) = Position(3)
