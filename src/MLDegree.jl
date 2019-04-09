module MLDegree

export GaussianML

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

Base.size(L::LogDet) = (div(L.n*(L.n+1), 2), div(L.n*(L.n+1), 2))

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



struct GaussianML{P<:SP.PolynomialSystem} <: HC.AbstractSystem
    F::P
    L::LogDet
end

function GaussianML(f::Vector{<:MP.AbstractPolynomialLike})
    F = SP.PolynomialSystem(f)
    m = length(f)
    n = round(Int, (sqrt(8m + 1) - 1) / 2)
    L = LogDet(n)
    GaussianML(F, L)
end

struct GaussianMLCache <: HC.AbstractSystemCache
    F_x::Vector{ComplexF64}
    J_F_x::Matrix{ComplexF64}
    H_F_x::Array{ComplexF64,3}
    log_det_u::Vector{ComplexF64}
    J_log_det_u::Matrix{ComplexF64}
    log_det_cache::LogDetCache
end

function HC.cache(GML::GaussianML, x, p)
    m, n = SP.npolynomials(GML.F), SP.nvariables(GML.F)
    F_x = zeros(ComplexF64, m)
    J_F_x = zeros(ComplexF64, m, n)
    H_F_x = zeros(ComplexF64, m, n, n)
    log_det_cache = LogDetCache(GML.L)
    log_det_u = zeros(ComplexF64, size(GML.L)[1])
    J_log_det_u = zeros(ComplexF64, size(GML.L)...)
    GaussianMLCache(F_x, J_F_x, H_F_x, log_det_u, J_log_det_u, log_det_cache)
end


function HC.evaluate!(u, GML::GaussianML, x, p, cache::GaussianMLCache)
    SP.evaluate_and_jacobian!(cache.F_x, cache.J_F_x, GML.F, x)
    evaluate!(cache.log_det_u, GML.L, cache.F_x, cache.log_det_cache)

    for k in 1:length(u)
        u[k] = zero(ComplexF64)
        # ∇_x log(det(K(x)))
        for i = 1:size(cache.J_F_x, 1)
            u[k] += cache.log_det_u[i] * cache.J_F_x[i, k]
        end

        # subtract now ∇_x trace(K(x)S)
        l = 1
        for i=1:GML.L.n, j in i:GML.L.n
            if i == j
                u[k] -=  cache.J_F_x[l, k] * p[l]
            else
                u[k] -= 2cache.J_F_x[l, k] * p[l]
            end
            l += 1
        end
    end

    u
end


function HC.jacobian!(U, GML::GaussianML, x, p, cache::GaussianMLCache)
    SP.evaluate!(cache.F_x, GML.F, x)
    SP.jacobian_and_hessian!(cache.J_F_x, cache.H_F_x, GML.F, x)
    evaluate_and_jacobian!(cache.log_det_u, cache.J_log_det_u, GML.L, cache.F_x, cache.log_det_cache)

    n = size(cache.J_F_x, 2)

    for j in 1:n, i in 1:n
        # Hessian_x log(det(K(x)))
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

        # subtract now Hessian_x trace(K(x)S)

        t = 1
        for k=1:GML.L.n, l in k:GML.L.n
            if k == l
                U[i, j] -=  cache.H_F_x[t, i, j] * p[t]
            else
                U[i, j] -= 2cache.H_F_x[t, i, j] * p[t]
            end
            t += 1
        end
    end
    U
end


function HC.evaluate_and_jacobian!(u, U, GML::GaussianML, x, p, cache::GaussianMLCache)
    SP.evaluate!(cache.F_x, GML.F, x)
    SP.jacobian_and_hessian!(cache.J_F_x, cache.H_F_x, GML.F, x)
    evaluate_and_jacobian!(cache.log_det_u, cache.J_log_det_u, GML.L, cache.F_x, cache.log_det_cache)

    for i in 1:length(u)
        # log(det(K(x)))
        u[i] = zero(ComplexF64)
        for k = 1:size(cache.J_F_x, 1)
            u[i] += cache.log_det_u[k] * cache.J_F_x[k, i]
        end

        # trace(K(x)S)
        l = 1
        for s=1:GML.L.n, t in s:GML.L.n
            if s == t
                u[i] -=  cache.J_F_x[l, i] * p[l]
            else
                u[i] -= 2cache.J_F_x[l, i] * p[l]
            end
            l += 1
        end
    end

    n = size(cache.J_F_x, 2)
    for j in 1:n, i in 1:n
        # log(det(K(x)))
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

        # subtract now Hessian_x trace(K(x)S)
        t = 1
        for k=1:GML.L.n, l in k:GML.L.n
            if k == l
                U[i, j] -=  cache.H_F_x[t, i, j] * p[t]
            else
                U[i, j] -= 2cache.H_F_x[t, i, j] * p[t]
            end
            t += 1
        end
    end

    nothing
end

function HC.differentiate_parameters!(U, GML::GaussianML, x, p, cache::GaussianMLCache)
    SP.jacobian!(cache.J_F_x, GML.F, x)
    for k in 1:size(U, 1)
        # compute ∂ / ∂s_{i,j} ∇_x -trace(K(x)S)
        l = 1
        for i=1:GML.L.n, j in i:GML.L.n
            if i == j
                U[k, l] =  -cache.J_F_x[l, k]
            else
                U[k, l] = -2cache.J_F_x[l, k]
            end
            l += 1
        end
    end
    U
end

end # module
