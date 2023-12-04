# Like:
# undef = UndefBlocksInitializer()
# undef_blocks = UndefBlocksInitializer()
abstract type AbstractInitializer end

struct ZeroInitializer <: AbstractInitializer end
const zero_init = ZeroInitializer()

# Equivalent to `Base.UndefUnitializer` and `Base.undef`,
# but a subtype of `AbstractInitializer`.
struct UndefInitializer <: AbstractInitializer end
const undef = UndefInitializer()

# TODO: Move to `AllocateDataBaseExt`.
base_initializer(::Base.UndefInitializer) = undef

# TODO: Move to `AllocateDataBlockArraysExt`.
using BlockArrays: BlockArrays

struct UndefBlocksInitializer <: AbstractInitializer end
const undef_blocks = UndefBlocksInitializer()

# TODO: Move to `AllocateDataBlockArraysExt`.
base_initializer(::BlockArrays.UndefBlocksInitializer) = BlockArrays.undef_blocks

# TODO: Add `rand_init`, `randn_init`?
