# Trait determining the style of inserting into a structure
abstract type InsertStyle end
struct IsInsertable <: InsertStyle end
struct NotInsertable <: InsertStyle end
struct FastCopy <: InsertStyle end

# Assume is insertable
@inline InsertStyle(::Type) = IsInsertable()
@inline InsertStyle(x) = InsertStyle(typeof(x))
