export AbstractModelContainer
export AbstractTurbulenceModel
export AbstractRANSModel, RANS
export AbstractLESModel, LES
export AbstractDESModel, DES

abstract type AbstractModelContainer end
abstract type AbstractTurbulenceModel end
abstract type AbstractBlendingMethod end
abstract type AbstractRANSModel <: AbstractTurbulenceModel end
abstract type AbstractLESModel <: AbstractTurbulenceModel end
abstract type AbstractDESModel <: AbstractTurbulenceModel end

Base.show(io::IO, model::AbstractTurbulenceModel) = print(io, typeof(model).name.wrapper)

# Models 
"""
    RANS <: AbstractRANSModel

Abstract RANS model type for consturcting RANS models.

### Fields
- 'args' -- Model arguments.
"""
struct RANS{T,ARG} <:AbstractModelContainer 
    args::ARG
end

"""
    LES <: AbstractLESModel

Abstract LES model type for constructing LES models.

### Fields
- 'args' -- Model arguments.
"""
struct LES{T,ARG} <:AbstractModelContainer 
    args::ARG
end

"""
    DES <: AbstractDESModel

Abstract DES model type for constructing DES models.

### Fields
- 'rans' -- The RANS model (Physics struct)
- 'les' -- The LES model (Physics struct)
- 'args' -- Model arguments.
"""
struct DES{T, ARG} <: AbstractModelContainer
    rans
    les
    args::ARG 
end
Adapt.@adapt_structure DES

struct MenterF1 <: AbstractBlendingMethod end
Adapt.@adapt_structure MenterF1