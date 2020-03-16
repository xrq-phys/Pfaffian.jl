"Module for skew-symmetric matrices."
module SkewSymmetric

using LinearAlgebra
using LinearAlgebra.BLAS: gemm!
using UnsafeArrays
using StaticArrays
# Alias
ᵀ = transpose
# No-allocation copy.
@inline cₚ!(M::MMatrix, A::Union{Array, SubArray}, δn, shiftn, δk, shiftk) = begin
    for l = 1:δk
        for i = 1:δn
            M[i, l] = A[shiftn+i, shiftk+l]
        end
    end
end
@inline wᵦ!(M::MMatrix, A::Union{Array, SubArray}, δn, shiftn, δk, shiftk) = begin
    for l = 1:δk
        for i = 1:δn
            A[shiftn+i, shiftk+l] = M[i, l]
        end
    end
end

"Currently only upper-triangular strict skew-symmetric matrices."
mutable struct AntiSymmetric{T} <: AbstractMatrix{T}
    M::Matrix{T}
    # M::UpperTriangular{T}
end

Base.size(C::AntiSymmetric) = size(C.M)
# AntiSymmetric(A::Matrix) = AntiSymmetric(UpperTriangular(A))

function Base.getindex(C::AntiSymmetric{T}, Id::Vararg{Int, 2}) where {T}
    i, j = Id
    if i == j
        return zero(eltype(C.M))
    elseif i < j
        return C.M[i, j]
    else
        return -C.M[j, i]
    end
end

function Base.setindex!(C::AntiSymmetric{T}, v::Number, Id::Vararg{Int, 2}) where {T}
    i, j = Id
    if i == j && abs(v) > eps(T)
        error("Trying to assign non-zero value to diagonal of skew-symmetric matrix.")
    elseif i < j
        C.M[i, j] = v
    else
        C.M[j, i] = -v
    end
end

function Base.broadcast(func, C::AntiSymmetric{T}, v::Number) where {T}
    D = AntiSymmetric{T}(similar(C.M))
    m, n = size(C)
    # Vector-like. No need to do blocking here.
    for j = 1:n
        for i = 1:j-1
            D.M[i, j] = func(C.M[i, j], v)
        end
    end
    D
end

function Base.broadcast!(func, C::AntiSymmetric{T}, v::Number) where {T}
    m, n = size(C)
    # Vector-like.
    for j = 1:n
        for i = 1:j-1
            C.M[i, j] = func(C.M[i, j], v)
        end
    end
    C
end

"Antisymmetric rank-2 update."
function r2!(C::AntiSymmetric, A::Vector, B::Vector, α::Number, β::Number)
    r2k!(C, reshape(A, length(A), 1), reshape(B, length(B), 1), α, β)
end

"Implements antisymmetric block rank-2 update: M + A B' - B A'"
function r2k!(C::AntiSymmetric, A::Matrix, B::Matrix, α::Number, β::Number; 
              Δn::Int=64, Δk::Int=64, νΔn::Int=4, νΔk::Int=8)
    if abs(α) ≤ eps(α)
        return broadcast!(*, C, β)
    end
    # Check parameters.
    n = size(A)[1]
    k, k_ = size(A)[2], size(B)[2]
    if k != k_
        error("SKR2K dimension mismatch.")
    end
    # Core scratchpad.
    ωA = @MMatrix zeros(νΔn, νΔk)
    ωB = @MMatrix zeros(νΔn, νΔk)
    ωC = @MMatrix zeros(νΔn, νΔn)
    # Blocked diagonal core
    @inline r2km!(νn, νM, νk, νA, νB, scale) = begin
        # Blocking scheme.
        # Exclusive.
        numνδn = νn ÷ νΔn
        νδfinn = νn % νΔn
        if νδfinn > 0
            numνδn += 1
        else
            νδfinn = νΔn
        end
        numνδk = νk ÷ νΔk
        νδfink = νk % νΔk
        if νδfink > 0
            numνδk += 1
        else
            νδfink = νΔk
        end
        @uviews νA begin 
        @uviews νB begin
        @uviews νM begin
        for jδ = 1:numνδn
            δj = if (jδ == numνδn) νδfinn
                else νΔn end
            shiftj = νΔn*(jδ-1)
            for lδ = 1:numνδk
                δl = if (lδ == numνδk) νδfink
                    else νΔk end
                shiftl = νΔk*(lδ-1)
                scaleν = scale && lδ == 1
                # B-blocking. Clear (only B) before copying.
                ωB .= zero(α)
                # ωB[1:δj, 1:δl] .= view(νB, (1:δj).+shiftj, (1:δl).+shiftl)
                cₚ!(ωB, νB, δj, shiftj, δl, shiftl)
                ωB .*= α
                for iδ = 1:numνδn
                    δi = if (iδ == numνδn) νδfinn
                        else νΔn end
                    shifti = νΔn*(iδ-1)
                    # A-blocking.
                    # ωA[1:δi, 1:δl] .= view(νA, (1:δi).+shifti, (1:δl).+shiftl)
                    cₚ!(ωA, νA, δi, shifti, δl, shiftl)
                    if iδ ≤ jδ
                        # C-caching.
                        # TODO: Copy explicitly to register?
                        # ωC[1:δi, 1:δj] .= view(νM, (1:δi).+shifti, (1:δj).+shiftj)
                        cₚ!(ωC, νM, δi, shifti, δj, shiftj)
                        if scaleν
                            ωC[1:δi, 1:δj] .*= β
                        end
                        if iδ == jδ
                            # Call vanilla microcore.
                            r2kμ!(δj, ωC,# view(νM, (1:δi).+shifti, (1:δj).+shiftj),
                                  δl, ωA, ωB, one(α), false)
                        else
                            # νM[(1:δi).+shifti, (1:δj).+shiftj] = begin
                            #     νM[(1:δi).+shifti, (1:δj).+shiftj] + (ωA* ᵀ(ωB))[1:δi, 1:δj]
                            # end
                            # mul!(ωC, ωA, ᵀ(ωB), one(α), one(β))
                            gemm!('N', 'T', one(α), ωA, ωB, one(β), ωC)
                        end
                        # νM[(1:δi).+shifti, (1:δj).+shiftj] .= view(ωC, 1:δi, 1:δj)
                        wᵦ!(ωC, νM, δi, shifti, δj, shiftj)
                    else
                        # ωC[1:δj, 1:δi] .= view(νM, (1:δj).+shiftj, (1:δi).+shifti)
                        cₚ!(ωC, νM, δj, shiftj, δi, shifti)
                        # mul!(ωC, ωB, ᵀ(ωA), -one(α), one(β))
                        gemm!('N', 'T', -one(α), ωB, ωA, one(β), ωC)
                        # νM[(1:δj).+shiftj, (1:δi).+shifti] = begin
                        #     νM[(1:δj).+shiftj, (1:δi).+shifti] + (ωB* ᵀ(ωA))[1:δj, 1:δi]
                        # end
                        # νM[(1:δj).+shiftj, (1:δi).+shifti] .= view(ωC, 1:δj, 1:δi)
                        wᵦ!(ωC, νM, δj, shiftj, δi, shifti)
                    end
                end
            end
        end
        end
        end
        end
    end
    # Vanilla microcore.
    @inline r2kμ!(μm, μM, μk, μA, μB, α_, scale) = begin
        for j = 1:μm
            if scale && abs(β) ≤ eps(β)
                μM[1:j-1, j] .= zero(μM[1])
            end
            if scale && abs(β-1) > eps(β)
                μM[1:j-1, j] .*= β
            end
            for l = 1:μk
                μarow = α_*μA[j, l]
                μbrow = α_*μB[j, l]
                if μarow != zero(β) || μbrow != zero(β)
                    for i = 1:j-1
                        μM[i, j] += μA[i, l]*μbrow - μB[i, l]*μarow
                    end
                end
            end
        end
    end
    # No blocking situation.
    if Δn < 4 || Δn > n
        Δn = n
    end
    if Δk < 4 || Δk > k
        Δk = k
    end
    # Prepare blocking.
    numδn = n ÷ Δn
    δfinn = n - Δn*(numδn-1)
    numδk = k ÷ Δk
    δfink = k - Δk*(numδk-1)

    # Launch the round-down blocking.
    CM = C.M
    @uviews A begin 
    @uviews B begin
    @uviews CM begin
    for jδ = 1:numδn
        δj = if (jδ == numδn) δfinn
            else Δn end
        shiftj = Δn*(jδ-1)
        for lδ = 1:numδk
            δl = if (lδ == numδk) δfink
                else Δk end
            scale = lδ == 1 && abs(β-1) > eps(β)
            shiftl = Δk*(lδ-1)
            for iδ = 1:numδn
                δi = if (iδ == numδn) δfinn
                    else Δn end
                shifti = Δn*(iδ-1)
                if iδ == jδ
                    # r2kμ!(δj, 
                    #       view(CM, (1:δi).+shifti, (1:δj).+shiftj), 
                    #       δl, 
                    #       view(A, (1:δi).+shifti, (1:δl).+shiftl),
                    #       view(B, (1:δj).+shiftj, (1:δl).+shiftl), α, scale)
                    r2km!(δj, 
                          view(CM, (1:δi).+shifti, (1:δj).+shiftj), 
                          δl, 
                          view(A, (1:δi).+shifti, (1:δl).+shiftl),
                          view(B, (1:δj).+shiftj, (1:δl).+shiftl), scale)
                else
                    if iδ < jδ
                        gemm!('N', 'T', α,
                              view(A, (1:δi).+shifti, (1:δl).+shiftl),
                              view(B, (1:δj).+shiftj, (1:δl).+shiftl),
                              if scale β 
                                  else one(β) end,
                              view(CM, (1:δi).+shifti, (1:δj).+shiftj))
                        # mul!(view(CM, (1:δi).+shifti, (1:δj).+shiftj),
                        #      view(A, (1:δi).+shifti, (1:δl).+shiftl),
                        #      ᵀ(view(B, (1:δj).+shiftj, (1:δl).+shiftl)), 
                        #      α, if scale β 
                        #          else one(β) end)
                    else
                        gemm!('N', 'T', -α,
                              view(B, (1:δj).+shiftj, (1:δl).+shiftl),
                              view(A, (1:δi).+shifti, (1:δl).+shiftl),
                              one(β),
                              view(CM, (1:δj).+shiftj, (1:δi).+shifti))
                        # mul!(view(CM, (1:δj).+shiftj, (1:δi).+shifti),
                        #      view(B, (1:δj).+shiftj, (1:δl).+shiftl),
                        #      ᵀ(view(A, (1:δi).+shifti, (1:δl).+shiftl)),
                        #      -α, one(β))
                    end
                end
            end
        end
    end
    end
    end
    end

    # Return reference.
    C
end

end
