using Plots
using XCALibre

mesh_file = "C:/Users/Hudson/OneDrive - The University of Nottingham/Year 3/Individual Project/Code/Meshes/2024-11-24-CircleMesh.unv"

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# Select backend and setup hardware
backend = CPU()
hardware = set_hardware(backend=backend, workgroup=4)

mesh_dev = mesh 

velocity = [4, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = LES{Smagorinsky}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall2, [0.0, 0.0, 0.0]),
    Wall(:wall1, [0.0, 0.0, 0.0]),
    Wall(:wall3, [0.0, 0.0, 0.0]), 
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall1, 0.0),
    Neumann(:wall2, 0.0),
    Neumann(:wall3, 0.0), 
)

schemes = (
    U = set_schemes(divergence = Linear),
    p = set_schemes() # no input provided (will use defaults)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # Options: GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), DILU(), ILU0()
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # Options: CgSolver, BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), LDL() (with GmresSolver)
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    )
)

# runtime = set_runtime(iterations=4000, time_step=1, write_interval=100)
runtime = set_runtime(iterations=1, time_step=1, write_interval=-1) # hide

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config);