export Laminar

# Model type definition (hold fields)
"""
    Laminar <: AbstractTurbulenceModel

Laminar model definition for physics API.
"""
struct Laminar <: AbstractRANSModel end 
Adapt.@adapt_structure Laminar

# Model type definition (hold equation definitions and internal data)
struct LaminarModel end 
Adapt.@adapt_structure LaminarModel

# Model API constructor (pass user input as keyword arguments and process if needed)
RANS{Laminar}() = begin # Empty constructor
    args = (); ARG = typeof(args)
    RANS{Laminar,ARG}(args)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(rans::RANS{Laminar, ARG})(mesh) where ARG = Laminar()

# Model initialisation
"""
    function initialise(
        turbulence::Laminar, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
        ) where {T,F,M,Tu,E,D,BI}
    return LaminarModel()
end

Initialisation of turbulent transport equations.

### Input
- `turbulence` -- turbulence model.
- `model`  -- Physics model defined by user.
- `mdtof`  -- Face mass flow.
- `peqn`   -- Pressure equation.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
          hardware structures set.

### Output
- `LaminarModel()`  -- Turbulence model structure.

"""
function initialise(
    turbulence::Laminar, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}
    return LaminarModel()
end

# Model solver call (implementation)

# Model solver call (implementation)
"""
    turbulence!(rans::LaminarModel, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time, config
    ) where {T,F,M,Tu<:Laminar,E,D,BI}

Run turbulence model transport equations.

### Input
- `rans::LaminarModel` -- Laminar turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `S2`  -- Square of the strain rate magnitude.
- `prev`  -- Previous field.
- `time`   -- 
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

"""
function turbulence!(rans::LaminarModel, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time, config
    ) where {T,F,M,Tu<:Laminar,E,D,BI}
    nothing
end

# Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, VTKWriter, name
    ) where {T,F,M,Tu<:Laminar,E,D,BI}
    if typeof(model.fluid)<:AbstractCompressible
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("T", model.energy.T)
        )
    else
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p)
        )
    end
    write_vtk(name, model.domain, VTKWriter, args...)
end

function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, VTKWriter, name
    ) where {T,F,M,Tu<:Laminar,E<:Nothing,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p),
    )
    write_vtk(name, model.domain, VTKWriter, args...)
end