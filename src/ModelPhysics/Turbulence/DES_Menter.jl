export MenterF1

#Model type definition (hold fields)
"""
    MenterF1 <: AbstractDESModel

Menter model containing all Menter field parameters

### Fields
- 'k' -- Turbulent kinetic energy ScalarField.
- 'omega' -- Specific dissipation rate ScalarField.
- 'nut' -- Eddy viscosity ScalarField.
- 'blnd_func' -- Menter blending function ScalarField.
- 'CDkw' -- Blending variable ScalarField
- 'kf' -- Turbulent kinetic energy FaceScalarField.
- 'omegaf' -- Specific dissipation rate FaceScalarField.
- 'nutf' -- Eddy viscosity FaceScalarField.
- 'coeffs' -- Model coefficients.
- 'rans' -- Stores the RANS model for blending.
- 'les' -- Stores the LES model for blending.
- 'y' -- Near-wall distance for model.
"""
struct MenterF1{S1,S2,S3,S4,S5,F1,F2,F3,C,M1,M2,Y} <: AbstractDESModel
    k::S1
    omega::S2
    nut::S3
    blnd_func::S4
    CDkw::S5
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
    rans::M1
    les::M2
    y::Y
    shield
end
Adapt.@adapt_structure MenterF1

struct MenterF1Model{M1,M2,E1,S1,S2,S3,V1,V2,State}
    ransTurbModel::M1
    lesTurbModel::M2
    ω_eqn::E1
    nueffω::S1
    Dωf::S2
    Pω::S3
    ∇k::V1
    ∇ω::V2
    state::State
end
Adapt.@adapt_structure MenterF1Model

#Model Constructor using a RANS and LES model
# Can be rewritten for K-ϵ model or another LES turbulence type
DES{MenterF1}(; RANSTurb=KOmega, LESTurb=Smagorinsky, walls,
    C_DES=0.65, σk1=0.85, σk2=1.00, σω1=0.65, σω2=0.856, β1=0.075, β2=0.0828, βstar=0.09, a1=0.31, β⁺=0.09, α1=0.52, σk=0.5, σω=0.5, C=0.15) = begin
    # Construct RANS turbulence
    rans = RANS{RANSTurb}()
    # Construct LES turbulence
    les = LES{LESTurb}()
    # DES coefficients
    args = (
        C_DES=C_DES,
        σk1=σk1,
        σk2=σk2,
        σω1=σω1,
        σω2=σω2,
        β1=β1,
        β2=β2,
        βstar=βstar,
        a1=a1,
        walls=walls,
        β⁺=β⁺,
        α1=α1,
        σk=σk,
        σω=σω,
        C=C
    )
    ARG = typeof(args)
    DES{MenterF1,ARG}(rans, les, args)
end


# Functor as constructor
(des::DES{MenterF1,ARG})(mesh) where {ARG} = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    blnd_func = ScalarField(mesh)
    CDkw = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = des.args
    rans = des.rans(mesh)
    les = des.les(mesh)
    shield = ScalarField(mesh)

    #create y values
    y = ScalarField(mesh)
    walls = des.args.walls
    BCs = []
    for boundary ∈ mesh.boundaries
        for namedwall ∈ walls
            if boundary.name == namedwall
                push!(BCs, Dirichlet(boundary.name, 0.0))
            else
                push!(BCs, Neumann(boundary.name, 0.0))
            end
        end
    end
    y = assign(y, BCs...)


    # δ = 0.01

    # function f(y)
    #     y = y/δ
    #     return -0.5 * (tanh(6*y-3.5)-1)
    # end
    
    # for (i,val) in enumerate(blnd_func.values)
    #     Cell = mesh.cells[i]
    #     ycell = Cell.centre[2]
    #     blnd_func.values[i] = f(ycell)
    # end


    MenterF1(k, omega, nut, blnd_func, CDkw, kf, omegaf, nutf, coeffs, rans, les, y, shield)
end

function initialise(turbulence::MenterF1, model::Physics, mdotf::FaceScalarField, p_eqn::ModelEquation, config)

    (; k, omega, nut, y, kf, omegaf, rans, les) = model.turbulence
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = p_eqn.equation

    nueffω = FaceScalarField(mesh)
    # nueffk = FaceScalarField(mesh)
    # Dkf = ScalarField(mesh)
    Dωf = ScalarField(mesh)
    dkdomegadx = ScalarField(mesh)
    # Pk = ScalarField(mesh)
    Pω = ScalarField(mesh)
    ∇k = Grad{schemes.k.gradient}(k)
    ∇ω = Grad{schemes.p.gradient}(omega)

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

    ransTurbModel = initialise(rans, model, mdotf, p_eqn, config)

    lesTurbModel = initialise(les, model, mdotf, p_eqn, config)


    init_residuals = (:k, 1.0), (:omega, 1.0)
    init_convergence = false
    state = ModelState(init_residuals, init_convergence)

    return MenterF1Model(ransTurbModel, lesTurbModel, ω_eqn, nueffω, Dωf, Pω, ∇k, ∇ω, state)

end

function turbulence!(
    des::MenterF1Model, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    (; rho) = model.fluid
    (; k, omega, nut, blnd_func, kf, omegaf, nutf, CDkw, rans, les, y, shield) = model.turbulence
    (; βstar, σω2) = model.turbulence.coeffs
    (; nueffω, Dωf, Pω, ω_eqn, ∇k, ∇ω) = des
    (; ransTurbModel, lesTurbModel) = des

    @. rans.nut.values = nut.values;
    @. les.nut.values = nut.values;

    turbulence!(ransTurbModel, model, S, prev, time, config)

    turbulence!(lesTurbModel, model, S, prev, time, config)

    nutRANS = rans.nut
    nutLES = les.nut

    # nueffω = get_flux(ω_eqn, 3)
    # Dωf = get_flux(ω_eqn, 4)
    # Pω = get_source(ω_eqn, 1)
    # dkdomegadx = get_source(ω_eqn, 2)

    # grad!(∇ω, omegaf, omega, omega.BCs, time, config)
    # grad!(∇k, kf, k, k.BCs, time, config)
    # inner_product!(dkdomegadx, ∇k, ∇ω, config)

    # @. CDkw.values = max((2 * rho.values * σω2 * (1 / omega.values) * dkdomegadx.values), 10e-20);
    # @. blnd_func.values = tanh(min(max(sqrt(k.values) / (βstar * y.values * omega.values),
    #  (500 * nut.values) / (y.values^2 * omega.values)),
    #  (4 * rho.values * σω2 * k.values) / (CDkw.values * y.values^2))^4);

    # @. blnd_func.values = ifelse(y.values < 0.02, 1.0, blnd_func.values);

    

    @. nut.values = (blnd_func.values * nutRANS.values) + ((1 - blnd_func.values) * nutLES.values)
    # @. nut.values = nutRANS.values;

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)

end

function turbulence!(
    rans::KOmegaModel{E1,E2,S1}, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
) where {T,F,M,Tu<:MenterF1,E,D,BI,E1,E2,S1}

    mesh = model.domain

    (; rho, rhof, nu, nuf) = model.fluid
    (; k, omega, kf, omegaf, coeffs) = model.turbulence
    (; nut, nutf) = model.turbulence.rans
    (; U, Uf, gradU) = S
    (; k_eqn, ω_eqn, state) = rans
    (; solvers, runtime) = config

    mueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    mueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)

    # update fluxes and sources

    # TO-DO: Need to bring gradient calculation inside turbulence models!!!!!

    grad!(gradU, Uf, U, U.BCs, time, config)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    magnitude2!(Pk, S, config, scale_factor=2.0) # multiplied by 2 (def of Sij)
    # constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only

    @. Pω.values = rho.values * coeffs.α1 * Pk.values
    @. Pk.values = rho.values * nut.values * Pk.values
    correct_production!(Pk, k.BCs, model, S.gradU, config) # Must be after previous line
    @. Dωf.values = rho.values * coeffs.β1 * omega.values
    @. mueffω.values = rhof.values * (nuf.values + coeffs.σω * nutf.values)
    @. Dkf.values = rho.values * coeffs.β⁺ * omega.values
    @. mueffk.values = rhof.values * (nuf.values + coeffs.σk * nutf.values)

    # Solve omega equation
    # prev .= omega.values
    discretise!(ω_eqn, omega, config)
    apply_boundary_conditions!(ω_eqn, omega.BCs, nothing, time, config)
    # implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    implicit_relaxation_diagdom!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    constrain_equation!(ω_eqn, omega.BCs, model, config) # active with WFs only
    update_preconditioner!(ω_eqn.preconditioner, mesh, config)
    ω_res = solve_system!(ω_eqn, solvers.omega, omega, nothing, config)

    # constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only
    bound!(omega, config)
    # explicit_relaxation!(omega, prev, solvers.omega.relax, config)

    # Solve k equation
    # prev .= k.values
    discretise!(k_eqn, k, config)
    apply_boundary_conditions!(k_eqn, k.BCs, nothing, time, config)
    # implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    implicit_relaxation_diagdom!(k_eqn, k.values, solvers.k.relax, nothing, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    k_res = solve_system!(k_eqn, solvers.k, k, nothing, config)
    bound!(k, config)
    # explicit_relaxation!(k, prev, solvers.k.relax, config)

    @. nut.values = k.values / omega.values

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)

    state.residuals = ((:k , k_res), (:omega, ω_res))
    state.converged = k_res < solvers.k.convergence && ω_res < solvers.omega.convergence
    return nothing
end

function turbulence!(
    les::SmagorinskyModel{E1,E2}, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
) where {T,F,M,Tu<:MenterF1,E,D,BI,E1,E2}

    mesh = model.domain

    (; coeffs) = model.turbulence
    (; nut, nutf) = model.turbulence.les
    (; U, Uf, gradU) = S
    (; Δ, magS) = les

    # grad!(gradU, Uf, U, U.BCs, time, config) # update gradient (internal structure of S)
    # limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    magnitude!(magS, S, config)
    @. magS.values *= sqrt(2) # should fuse into definition of magnitude function!

    # update eddy viscosity 
    @. nut.values = coeffs.C * Δ.values * magS.values # careful: here Δ = Δ²

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)
end

#Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, VTKWriter, name
) where {T,F,M,Tu<:MenterF1,E,D,BI}
    if typeof(model.fluid) <: AbstractCompressible
        args = (
            ("U", model.momentum.U),
            ("p", model.momentum.p),
            ("T", model.energy.T),
            ("k", model.turbulence.k),
            ("omega", model.turbulence.omega),
            ("nut", model.turbulence.nut)
        )
    else
        args = (
            ("U", model.momentum.U),
            ("p", model.momentum.p),
            ("k", model.turbulence.k),
            ("omega", model.turbulence.omega),
            ("nut", model.turbulence.nut),
            ("y", model.turbulence.y),
            ("F1", model.turbulence.blnd_func),
            ("ransnut", model.turbulence.rans.nut),
            ("lesnut", model.turbulence.les.nut),
            ("Shield", model.turbulence.shield)
        )
    end
    write_vtk(name, model.domain, VTKWriter, args...)
end