using MLDegree
using DynamicPolynomials, Test
using LinearAlgebra

@testset "MLDegree" begin
    @polyvar p[1:3] s[1:6]
    vars = p
    model = [p[1]+p[2], p[3]^2, p[1]*p[3], p[2]+p[3], (p[1]+p[3])^2, p[1]+p[3]]

    K = [p[1]+p[2] p[3]^2         p[1]*p[3]
         p[3]^2    p[2]+p[3]      (p[1]+p[3])^2
         p[1]*p[3] (p[1]+p[3])^2   p[1]+p[3]]
    S = [s[1] s[2] s[3]; s[2] s[4] s[5]; s[3] s[5] s[6]]

    det_K = det(K)
    ∇log_det_K = differentiate(det_K, vars) ./ det_K
    J_log_det_K = differentiate(∇log_det_K, vars)

    ∇trace_K_S = differentiate(tr(K*S), vars)
    J_trace_K_S = differentiate(∇trace_K_S, vars)

    ml = ∇log_det_K - ∇trace_K_S
    J_ml = J_log_det_K - J_trace_K_S


    u = zeros(ComplexF64, length(vars))
    U = zeros(ComplexF64, length(vars), length(vars))
    w = rand(ComplexF64, length(vars))
    p = rand(6)

    PLD = MLDegree.GaussianML(model)
    cache = HC.cache(PLD, w, S)

    @test HC.evaluate!(u, PLD, w, p, cache) ≈ map(f -> f(vars=>w, s=>p), ml) atol=1e-12
    @test HC.jacobian!(U, PLD, w, p, cache) ≈ map(f -> f(vars=>w, s=>p), J_ml) atol=1e-12

    u .= 0
    U .= 0
    HC.evaluate_and_jacobian!(u, U, PLD, w, p, cache)
    @test u ≈ map(f -> f(vars=>w, s=>p), ml) atol=1e-12
    @test U ≈ map(f -> f(vars=>w, s=>p), J_ml) atol=1e-12
end
