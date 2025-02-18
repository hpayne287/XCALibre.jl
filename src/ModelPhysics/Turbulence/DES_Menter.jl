export MenterF1

#Model type definition (hold fields)
"""
    Menter <: AbstractDESModel

Menter model containing all Menter field parameters

### Fields
- 'k' -- Turbulent kinetic energy ScalarField.
- 'omega' -- Specific dissipation rate ScalarField.
- 'nut' -- Eddy viscosity ScalarField.
- 'blnd_func1' -- Menter blending function ScalarField.
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
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
    rans::M1
    les::M2
    y::Y
end
Adapt.@adapt_structure MenterF1

struct MenterF1Model{E1,E2,E3,S1,V1,V2,State}
    k_eqn::E1
    ω_eqn::E2
    F1_eqn::E3
    nuts::S1
    ∇k::V1
    ∇ω::V2
    state::State
end
Adapt.@adapt_structure MenterF1Model

#Model Constructor using a RANS and LES model
# Can be rewritten for K-ϵ model or another LES turbulence type
DES{MenterF1}(mesh_dev, nu; RANSTurb=KOmega, LESTurb=Smagorinsky, walls,
    C_DES=0.65, σk1=0.85, σk2=1.00, σω1=0.65, σω2=0.856, β1=0.075, β2=0.0828, βstar=0.09, a1=0.31) = begin
    # Construct RANS model
    rans = Physics(
        time=Transient(),
        fluid=Fluid{Incompressible}(nu=nu),
        turbulence=RANS{RANSTurb}(),
        energy=Energy{Isothermal}(),
        domain=mesh_dev
    )
    # Construct LES model
    les = Physics(
        time=Transient(),
        fluid=Fluid{Incompressible}(nu=nu),
        turbulence=LES{LESTurb}(),
        energy=Energy{Isothermal}(),
        domain=mesh_dev
    )
    # DES coefficients
    des_args = (
        C_DES=C_DES,
        σk1=σk1,
        σk2=σk2,
        σω1=σω1,
        σω2=σω2,
        β1=β1,
        β2=β2,
        βstar=βstar,
        a1=a1,
        walls=walls)
    ARG = typeof(des_args)
    DES{MenterF1,ARG}(rans, les, des_args)
end


# Functor as constructor
(des::DES{MenterF1,ARG})(mesh) where {ARG} = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    blnd_func = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    momentum = Momentum(mesh)
    coeffs = des.args
    rans = des.rans
    les = des.les

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
    # ranscoeffs = des.rans.turbulence.coeffs
    # lescoeffs = des.les.turbulence.coeff
    # KOmega(k,omega,nut,kf,omegaf,nutf,ranscoeffs) #These are already made within the rans and les model no?
    # Smagorinsky(nut,nutf,lescoeffs)
    MenterF1(k, omega, nut, blnd_func, kf, omegaf, nutf, coeffs, rans, les, y)
end

function initialise(turbulence::MenterF1, model::Physics, mdotf::FaceScalarField, peqn::ModelEquation, config)

    (; k, omega, nut, y, kf, omegaf, βstar, σω2) = model.turbulence
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    ∇k = Grad{schemes.k.gradient}(k)
    ∇ω = Grad{schemes.p.gradient}(omega)

    k_eqn = (
        Time{schemes.k.time}(k)
        +
        Divergence{schemes.k.divergence}(mdotf, k)
        -
        Laplacian{schemes.k.laplacian}(nueffk, k)
        +
        Si(Dkf, k) # Dkf = β⁺*omega
        ==
        Source(Pk)
    ) → eqn

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

    F1_eqn = (
        tanh(min(max(sqrt(k) / (βstar * y), (500 * nut) / (y^2 * omega)), (4 * model.fluid.rho * σω2 * k) / (CDkw * y^2))^4)
    ) → eqn

    wall_distance!(model, config)

    init_residuals = (:k, 1.0), (:omega, 1.0)
    init_convergence = false
    state = ModelState(init_residuals, init_convergence)

    return MenterF1Model(k_eqn,ω_eqn,F1_eqn,nuts,∇k,∇ω,state)

end

function turbulence!(des::DES, model, mesh, config)

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
            ("nut", model.turbulence.nut)
        )
    end
    write_vtk(name, model.domain, VTKWriter, args...)
end