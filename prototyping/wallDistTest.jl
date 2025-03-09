using XCALibre
# using CUDA

# backwardFacingStep_2mm, backwardFacingStep_10mm
# mesh_file = "unv_sample_meshes/flatplate_transition.unv"
# mesh_file = "unv_sample_meshes/flatplate_2D_lowRe.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_lowRe.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=1024)

mesh_dev = adapt(hardware.backend, mesh)

# Turbulence Model
velocity = [5.4,0,0]
nu = 1.48e-5
Re = 10*1/nu
νR = 15
Tu = 0.03
k_inlet = 0.0575 #3/2*(Tu*velocity[1])^2
kL_inlet = 0.0115 #1/2*(Tu*velocity[1])^2
ω_inlet = 275 #k_inlet/(νR*nu)

# model = RANS{KOmegaLKE}(mesh=mesh, viscosity=nu, Tu=Tu)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmegaLKE}(Tu = 0.01, walls=(:wall,)),
    energy = Energy{Isothermal}(),
    domain = mesh
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Neumann(:top, 0.0),
    # Dirichlet(:bottom, velocity),
    # Neumann(:freestream, 0.0),
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0),
    # Neumann(:bottom, 0.0),
    # Neumann(:freestream, 0.0)
)

@assign! model turbulence kl (
    Dirichlet(:inlet, kL_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 1e-15),
    Neumann(:top, 0.0),
    # Neumann(:bottom, 0.0),
    # Neumann(:freestream, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Neumann(:top, 0.0),
    # Neumann(:bottom, 0.0),
    # Neumann(:freestream, 0.0)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    Neumann(:top, 0.0),
    # Neumann(:bottom, 0.0),
    # Neumann(:freestream, 0.0)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0), 
    Neumann(:top, 0.0),
    # Neumann(:bottom, 0.0),
    # Neumann(:freestream, 0.0),
)

schemes = (
    U = set_schemes(divergence=LUST),
    p = set_schemes(divergence=Upwind),
    k = set_schemes(divergence=Upwind),
    y = set_schemes(gradient=Midpoint),
    kl = set_schemes(divergence=Upwind,gradient=Midpoint),
    omega = set_schemes(divergence=Upwind)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.7,
        rtol = 1e-2,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver, CgSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.2,
        rtol = 1e-3,
    ),
    y = set_solver(
        model.turbulence.y;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.9,
    ),
    kl = set_solver(
        model.turbulence.kl;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    )
)

runtime = set_runtime(
    iterations=1, write_interval=1, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.kl, kL_inlet)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config); #, pref=0.0) # 9.39k allocs
