export Hybrid

#Model type definition (hold fields)
"""
    Hybrid <: AbstractDESModel

Hybrid model containing all Hybrid field parameters

### Fields
- `k` -- Turbulent kinetic energy ScalarField.
- `omega` -- Specific dissipation rate ScalarField.
- `nut` -- Eddy viscosity ScalarField.
- `blendWeight` -- blending weights ScalarField.
- `CDkw` -- Blending variable ScalarField
- `kf` -- Turbulent kinetic energy FaceScalarField.
- `omegaf` -- Specific dissipation rate FaceScalarField.
- `nutf` -- Eddy viscosity FaceScalarField.
- `coeffs` -- Model coefficients.
- `rans` -- Stores the RANS model for blending.
- `les` -- Stores the LES model for blending.
- `y` -- Wall normal distance for model.
"""
struct Hybrid{S1,S2,S3,S4,S5,F1,F2,F3,C,M1,M2,Y} <: AbstractDESModel
    k::S1
    omega::S2
    nut::S3
    blendWeight::S4
    CDkw::S5
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
    rans::M1
    les::M2
    y::Y
end
Adapt.@adapt_structure Hybrid

struct HybridModel{M1,M2,E1,S1,S2,S3,V1,D,V2,State}
    ransModel::M1
    lesModel::M2
    ω_eqn::E1
    nueffω::S1
    Dωf::S2
    Pω::S3
    ∇k::V1
    Δ::D
    ∇ω::V2
    state::State
end
Adapt.@adapt_structure HybridModel

#Model Constructor 
"""
    DES{Hybrid}(; TurbModel1=RANS, Turb1=KOmega, TurbModel2=LES, Turb2=Smagorinsky, blendType=MenterF1(), walls, args)

Contruct a hybrid model

### Inputs
- `TurbModel1` -- Type of the first model, defaults to RANS
- `TurbModel2` -- Type of the second model, defaults to LES
- `Turb1` -- Turbulence model to be used in first model, defaults to KOmega
- `Turb2` -- Turbulence model to be used in second model, defaults to Smagorinsky
- `blendType` -- Blending method to be used, defaults to MenterF1
- `walls` -- Required field holding all wall boundaries
- Will also take values for any coefficients, all have default values
"""
DES{Hybrid}(; TurbModel1=RANS, Turb1=KOmega, TurbModel2=LES, Turb2=Smagorinsky, blendType=MenterF1(), walls,
    C_DES=0.65, σk1=0.85, σk2=1.00, σω1=0.65, σω2=0.856, σd=0.125, β1=0.075, β2=0.0828, βstar=0.09, a1=0.31, β⁺=0.09, α1=0.52, σk=0.5, σω=0.5, C=0.15) = begin
    # Construct RANS turbulence
    rans = TurbModel1{Turb1}()
    # Construct LES turbulence
    les = TurbModel2{Turb2}()
    #DES coefficients 
    #THIS SHOULD BE THINNED OUT AFTER HP/FieldMoving is applied, each turbulenceModel will store their own kw args, specifying them might be hard though
    args = (
        C_DES=C_DES,
        σk1=σk1,
        σk2=σk2,
        σω1=σω1,
        σω2=σω2,
        σd=σd,
        β1=β1,
        β2=β2,
        βstar=βstar,
        a1=a1,
        walls=walls,
        β⁺=β⁺,
        α1=α1,
        σk=σk,
        σω=σω,
        C=C,
        blendType=blendType
    )
    ARG = typeof(args)
    DES{Hybrid,ARG}(rans, les, args)
end


# Functor as constructor
(des::DES{Hybrid,ARG})(mesh) where {ARG} = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    blendWeight = ScalarField(mesh)
    CDkw = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = des.args
    rans = des.rans(mesh)
    les = des.les(mesh)
    y = ScalarField(mesh)


    # Allocate wall distance "y" and setup boundary conditions
    walls = des.args.walls
    boundaries_cpu = get_boundaries(mesh.boundaries)
    BCs = []
    for boundary ∈ boundaries_cpu
        for namedwall ∈ walls
            if boundary.name == namedwall
                push!(BCs, Dirichlet(boundary.name, 0.0))
            else
                push!(BCs, Wall(boundary.name, 0.0))
            end
        end
    end
    y = assign(y, BCs...)

    Hybrid(k, omega, nut, blendWeight, CDkw, kf, omegaf, nutf, coeffs, rans, les, y)
end

#Model initialisation
"""
    initialise(turbulence::Hybrid, model::Physics, mdotf::FaceScalarField, p_eqn::ModelEquation, config)

Initialisation of turbulent transport equations 

### Input
- `turbulence` -- turbulence model.
- `model`  -- Physics model defined by user.
- `mdtof`  -- Face mass flow.
- `peqn`   -- Pressure equation.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
          hardware structures set.

### Output
- `HybridModel(
        ransModel,
        lesModel,
        ω_eqn,
        nueffω,
        Dωf,
        Pω,
        ∇k,
        ∇ω,
        state
    )` -- Turblence model structure

"""
function initialise(turbulence::Hybrid, model::Physics, mdotf::FaceScalarField, p_eqn::ModelEquation, config)

    @info "Initialising Hybrid Framework..."

    (; k, omega, nut, y, kf, omegaf, rans, les) = model.turbulence
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = p_eqn.equation

    nueffω = FaceScalarField(mesh)
    Dωf = ScalarField(mesh)
    dkdomegadx = ScalarField(mesh)
    Pω = ScalarField(mesh)
    ∇k = Grad{schemes.k.gradient}(k)
    ∇ω = Grad{schemes.p.gradient}(omega)
    Δ = ScalarField(mesh)
    delta!(Δ, mesh, config)

    ω_eqn = (
        Time{schemes.omega.time}(omega)
        +
        Divergence{schemes.omega.divergence}(mdotf, omega)
        -
        Laplacian{schemes.omega.laplacian}(nueffω, omega)
        +
        Si(Dωf, omega)  # Dωf = β1*omega
        ==
        Source(Pω)
        +
        Source(dkdomegadx)
    ) → eqn

    @reset ω_eqn.preconditioner = set_preconditioner(
        solvers.omega.preconditioner, ω_eqn, omega.BCs, config)


    @reset ω_eqn.solver = solvers.omega.solver(_A(ω_eqn), _b(ω_eqn))

    TF = _get_float(mesh)
    time = zero(TF) # assuming time=0
    grad!(∇ω, omegaf, omega, omega.BCs, time, config)
    grad!(∇k, kf, k, k.BCs, time, config)

    wall_distance!(model, config)

    #Create Turbulence models
    ransModel = initialise(rans, model, mdotf, p_eqn, config)
    lesModel = initialise(les, model, mdotf, p_eqn, config)

    @. rans.k.values = k.values
    @. rans.omega.values = omega.values

    init_residuals = (:k, 1.0), (:omega, 1.0)
    init_convergence = false
    state = ModelState(init_residuals, init_convergence)

    set_eddy_viscosity(model)
    
    return HybridModel(
        ransModel,
        lesModel,
        ω_eqn,
        nueffω,
        Dωf,
        Pω,
        ∇k,
        Δ,
        ∇ω,
        state
    )

end

#Model solver call (implementation)
"""
    turbulence!(
    des::HybridModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Run turbulence model transport equations.

### Input
- `des::HybridModel` -- Hybrid turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `prev`  -- Previous field.
- `time`   -- 
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.
"""
function turbulence!(
    des::HybridModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    (; nut, blendWeight, nutf, rans, les) = model.turbulence
    (; blendType) = model.turbulence.coeffs
    (; ransModel, lesModel) = des

    set_eddy_viscosity(model)

    turbulence!(ransModel, model, S, prev, time, config)
    turbulence!(lesModel, model, S, prev, time, config)

    update_model_parameters!(model)
    update_blend_weights!(blendType,des,model,config) 

    blend_nut!(nut, blendWeight, rans.nut, les.nut)

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)
end

#Specialise output writer
function save_output(model::Physics{T,F,M,Tu,E,D,BI}, outputWriter, iteration
) where {T,F,M,Tu<:Hybrid,E,D,BI}

    args = (
        ("U", model.momentum.U),
        ("p", model.momentum.p),
        ("k", model.turbulence.k),
        ("omega", model.turbulence.omega),
        ("nut", model.turbulence.nut),
        ("y", model.turbulence.y),
        ("blending_function", model.turbulence.blendWeight)
    )
    write_results(iteration, model.domain, outputWriter, args...)
end