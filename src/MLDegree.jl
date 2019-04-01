module MLDegree

import StaticPolynomials, HomotopyContinuation
import LinearAlgebra
import MultivariatePolynomials

const MP = MultivariatePolynomials
const SP = StaticPolynomials
const HC = HomotopyContinuation


"""
    LogDet(n)

Compute with `d / d_x_{ij} log(det(X))` where `X` is a symmetric n×n matrix.
"""
struct LogDet
    n::Int
end

struct LogDetCache
    lufac::LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}}
    X::Matrix{ComplexF64}
end

function LogDetCache(F::LogDet)
    X = randn(ComplexF64, F.n, F.n)
    lufac = LinearAlgebra.lu(X)
    LogDetCache(lufac, X)
end

function compute_matrix_inv!(cache::LogDetCache, x::AbstractVector)
    construct_symmetric!(cache.X, x)
    compute_matrix_inv!(cache)
end
function compute_matrix_inv!(cache::LogDetCache)
    HC.factorize!(cache.lufac, cache.X)
    LinearAlgebra.inv!(cache.lufac)
end


function construct_symmetric!(X, x)
    n = size(X, 1)
    k = 1
    for i=1:n
        for j=i:n
            X[i,j] = X[j, i] = x[k]
            k += 1
        end
    end
    X
end

function evaluate!(u, S::LogDet, x, cache::LogDetCache)
    n = size(cache.X, 1)
    X⁻¹ = compute_matrix_inv!(cache, x)
    k = 1
    for i in 1:n, j in i:n
        if i == j
            u[k] = X⁻¹[i,i]
        else
            u[k] = 2X⁻¹[i, j]
        end
        k += 1
    end

    u
end

function jacobian!(U, S::LogDet, x, cache::LogDetCache)
    X = cache.X
    n = size(X, 1)
    X⁻¹ = compute_matrix_inv!(cache, x)

    s = 1
    for i in 1:n, j in i:n
        t = 1
        for k in 1:n, l in k:n
            if i != j
                U[s, t] = 2
            else
                U[s, t] = 1
            end
            if k == l
                U[s, t] *= X⁻¹[i, l] * X⁻¹[k, j]
            else
                U[s, t] *= (X⁻¹[i, l] * X⁻¹[k, j] + X⁻¹[i, k] * X⁻¹[l, j])
            end
            t += 1
        end
        s += 1
    end

    U
end

function evaluate_and_jacobian!(u, U, S::LogDet, x, cache::LogDetCache)
    X = cache.X
    n = size(X, 1)
    X⁻¹ = compute_matrix_inv!(cache, x)

    k = 1
    for i in 1:n, j in i:n
        if i == j
            u[k] = X⁻¹[i,i]
        else
            u[k] = 2X⁻¹[i, j]
        end
        k += 1
    end
    s = 1
    for i in 1:n, j in i:n
        t = 1
        for k in 1:n, l in k:n
            if i != j
                U[s, t] = 2
            else
                U[s, t] = 1
            end
            if k == l
                U[s, t] *= X⁻¹[i, l] * X⁻¹[k, j]
            else
                U[s, t] *= (X⁻¹[i, l] * X⁻¹[k, j] + X⁻¹[i, k] * X⁻¹[l, j])
            end
            t += 1
        end
        s += 1
    end

    nothing
end



struct ParameterizedLogDet{P<:SP.PolynomialSystem}
    F::P
    L::LogDet
end

function ParameterizedLogDet(f::Vector{<:MP.Abstract.PolynomialLike})
    F = SP.PolynomialSystem(f)
    m = length(f)
    n = div(round(Int, sqrt(8m + 1) + 1), 2)
    L = LogDet(n)
    ParameterizedLogDet(F, L)
end

struct ParameterizedLogDetCache
    F_x::Vector{ComplexF64}
    J_F_x::Matrix{ComplexF64}
    H_F_x::Array{ComplexF64,3}
    log_det_u::Vector{ComplexF64}
    J_log_det_u::Matrix{ComplexF64}
    log_det_cache::LogDetCache
end

function ParameterizedLogDetCache(S::ParameterizedLogDet, x)
    m, n = SP.npolynomials(S.F), SP.nvariables(S.F)
    F_x = zeros(ComplexF64, m)
    J_F_x = zeros(ComplexF64, m, n)
    H_F_x = zeros(ComplexF64, m, n, n)
    log_det_cache = LogDetCache(S.L)
    log_det_u = zeros(ComplexF64, div(m * (m - 1), 2))
    J_log_det_u = zeros(ComplexF64, div(m * (m - 1), 2), div(m * (m - 1), 2))
    ParameterizedLogDetCache(F_x, J_F_x, H_F_x, log_det_u, J_log_det_u, log_det_cache)
end


function evaluate!(u, S::ParameterizedLogDet, x, cache::ParameterizedLogDetCache)
    SP.evaluate_and_jacobian!(cache.F_x, cache.J_F_x, S.F, x)
    evaluate!(cache.log_det_u, S.L, cache.F_x, cache.log_det_cache)

    for i in 1:length(u)
        u[i] = zero(ComplexF64)
        for k = 1:size(cache.J_F_x, 1)
            u[i] += cache.log_det_u[k] * cache.J_F_x[k, i]
        end
    end
    u
end


function jacobian!(U, S::ParameterizedLogDet, x, cache::ParameterizedLogDetCache)
    SP.evaluate!(cache.F_x, S.F, x)
    SP.jacobian_and_hessian!(cache.J_F_x, cache.H_F_x, S.F, x)
    evaluate_and_jacobian!(cache.log_det_u, cache.J_log_det_u, S.L, cache.F_x, cache.log_det_cache)

    n = size(cache.J_F_x, 2)

    for j in 1:n, i in 1:n
        U[i, j] = zero(ComplexF64)
        for k = 1:size(cache.J_F_x, 1)
            U[i, j] += cache.log_det_u[k] * cache.H_F_x[k, i, j]
        end

        for l = 1:size(cache.J_F_x, 1)
            U_i_j_k = zero(eltype(U))
            for k = 1:size(cache.J_F_x, 1)
                U_i_j_k += cache.J_log_det_u[k, l] * cache.J_F_x[k, i]
            end
            U[i, j] -= U_i_j_k * cache.J_F_x[l, j]
        end
    end
    U
end


function evaluate_and_jacobian!(u, U, S::ParameterizedLogDet, x, cache::ParameterizedLogDetCache)
    SP.evaluate!(cache.F_x, S.F, x)
    SP.jacobian_and_hessian!(cache.J_F_x, cache.H_F_x, S.F, x)
    evaluate_and_jacobian!(cache.log_det_u, cache.J_log_det_u, S.L, cache.F_x, cache.log_det_cache)

    for i in 1:length(u)
        u[i] = zero(ComplexF64)
        for k = 1:size(cache.J_F_x, 1)
            u[i] += cache.log_det_u[k] * cache.J_F_x[k, i]
        end
    end

    n = size(cache.J_F_x, 2)
    for j in 1:n, i in 1:n
        U[i, j] = zero(ComplexF64)
        for k = 1:size(cache.J_F_x, 1)
            U[i, j] += cache.log_det_u[k] * cache.H_F_x[k, i, j]
        end

        for l = 1:size(cache.J_F_x, 1)
            U_i_j_k = zero(eltype(U))
            for k = 1:size(cache.J_F_x, 1)
                U_i_j_k += cache.J_log_det_u[k, l] * cache.J_F_x[k, i]
            end
            U[i, j] -= U_i_j_k * cache.J_F_x[l, j]
        end
    end

    nothing
end


end # module
