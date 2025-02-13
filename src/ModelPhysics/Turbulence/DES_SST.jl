export Menter

struct Menter{S1,S2,S3,F1,F2,F3,C,M1,M2,Y} <: AbstractDESModel
    k::S1
    omega::S2
    nut::S3
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
    rans::M1
    les::M2
    y::Y
end
Adapt.@adapt_structure Menter

#Model Constructor using a RANS and LES model
# Can be rewritten for K-ϵ model or another LES turbulence type
DES{Menter}(mesh_dev,nu;RANSTurb,LESTurb,walls) = begin
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
        C_DES=0.65, 
        σk1=0.85, 
        σk2=1.00,
        σω1=0.65,
        σω2=0.856,
        β1=0.075,
        β2=0.0828,
        βstar=0.09,
        a1=0.31,
        walls=walls) 
    ARG = typeof(des_args)
    DES{Menter,ARG}(rans, les, des_args)
end


# Functor as constructor
(des::DES{Menter,ARG})(mesh) where {ARG} = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    momentum=Momentum(mesh)
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

    KOmega()
    Smagorinksy()
end

function initialise(T::Menter, model::Physics, mdotf::FaceScalarField, peqn::ModelEquation, config)

    #This feels like the wrong way of storing access to the LES and RANS models but works for now
    rans = T.rans_model
    les = T.les_model
    velocity = T.args.v
    # Initialise RANS model 
    initialise!(rans.momentum.U, velocity) #Need to review what actually gets passed to the RANS and LES initialise methods


    # Initialise LES model 
    initialise(les.turbulence, les, mdotf, peqn, config)

    # Compute initial DES length scale
    des_length_scale = compute_DES_lengthscale(rans, les, mesh)

    # Store the length scale inside the DES model 
    des.args = merge(des.args, (L_DES=des_length_scale,))
end

function turbulence!(des::DES, model, mesh, config)
    ν_t = compute_turbulent_viscosity(des, model, mesh)
    apply_turbulence_model!(des, model, ν_t, config)
end

#Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, VTKWriter, name
) where {T,F,M,Tu<:Menter,E,D,BI}
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