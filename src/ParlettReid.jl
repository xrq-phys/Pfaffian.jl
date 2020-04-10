# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"Module for Parlett-Reid transformation."
module ParlettReid

using UnsafeArrays
using StaticArrays
using LinearAlgebra
using LinearAlgebra.BLAS: ger!, gemm!
using ..SkewSymmetric: AntiSymmetric, r2k!, r2!, ᵀ

function inv_tile(A::AntiSymmetric{T}) where {T}
    n₁, n₂ = size(A)

    # TODO: change to j->i.
    # TODO: improve locality.
    for i = 1:2:n₁-1
        local r = 1.0
        for j = i+1:2:n₂
            r /= A[j-1, j] # aᵢ
            A[i, j] = -r
            if j ≠ n₂
                r *= A[j, j+1] # bᵢ
            end
        end
        if i ≠ n₁-1
            # Zeroize next line.
            A[i+1, i+2] = 0.0
        end
    end
    A
end

"""
    simple(A::AntiSymmetric{T})

Finds Pfaffian and inverse without pivoting.
Not safe. Only use it for debugging.
"""
function simple(A::AntiSymmetric{T}; calc_inv::Bool=false) where {T}
    n₁, n₂ = size(A)

    local M = zero(A) + I # which is actually Mᵀ
    local aₖ = A[:, 1]
    for i = 1:n₁-2
        αₖ = aₖ # A[:, i], this is subscripted i+1 according to Wimmer's paper.
        # keep aₖ to next step,
        # spoiling property that A[:, i+1] doesn't change after each update.
        aₖ = A[:, i+1]
        αₖ /= αₖ[i+1]
        αₖ[1:i+1] .= 0.0
        r2!(A, αₖ, aₖ, 1.0, 1.0)
        if calc_inv
            # mₖ = M[i+1, :] for original M
            mₖ = M[:, i+1]
            # M -= αₖ*mₖ' for original M
            # M -= mₖ*αₖ', which is:
            ger!(-1.0, mₖ, αₖ, M)
        end
    end; 
    PfA = prod([A[i, i+1] for i=1:2:n₁-1])
    if calc_inv
        PfA, M*inv_tile(A)* ᵀ(M)
    else
        (PfA, )
    end
end

"Blocked version of `ParlettReid.simple`."
function blocked(A::AntiSymmetric{T}; calc_inv::Bool=false, block_size::Int=0) where {T}
    n₁, n₂ = size(A)
    if block_size < 1 || block_size > n₁
        return simple(A, calc_inv=calc_inv)
    end
    # Function for view-swapping.
    swapln(v1::SubArray{T, 1}, v2::SubArray{T, 1}, cnt::Int, sign::Bool=false) = begin
        if !sign
            for i = 1:cnt
                telem = v1[i]
                v1[i] = v2[i]
                v2[i] = telem
            end
        else
            for i = 1:cnt
                telem = v1[i]
                v1[i] =-v2[i]
                v2[i] =-telem
            end
        end
    end

    local M = zero(A) + I
    vV = @MMatrix zeros(n₁, block_size)
    vW = @MMatrix zeros(n₁, block_size)
    mW = @MMatrix zeros(n₁, block_size) # changes in M.
    mM = @MMatrix zeros(n₁, block_size) # for slicing M.
    @uviews vV begin
    @uviews vW begin
    @uviews mW begin
    @uviews mM begin
    for ist = 1:block_size:n₁-2
        δi = if (ist+block_size-1 > n₁-2) n₁ - ist - 1
            else block_size end
        # A contribution to uₖ.
        local aₖ = A[:, ist]
        αₖ = @MVector zeros(n₁)
        mₖ = @MVector zeros(block_size)
        for i = 0:δi-1
            icur = ist+i
            αₖ .= aₖ # updated A[:, icur].
            if abs(αₖ[icur+1]) < 1e-4
                # Pivoting
                sₐ = icur+1
                α₁, tₐ = findmax(abs.(αₖ))
                α₁ = αₖ[tₐ] # use value instead of abs.
                if sₐ < tₐ
                    # Swap the updates.
                    if i != 0
                        swapln(view(vV, sₐ, 1:δi), view(vV, tₐ, 1:δi), δi)
                        swapln(view(vW, sₐ, 1:δi), view(vW, tₐ, 1:δi), δi)
                    end
                    # Swapping for A.
                    swapln(view(A, 1:sₐ-1, sₐ), view(A, 1:sₐ-1, tₐ), sₐ-1)
                    A[sₐ, tₐ] *= -1
                    if tₐ > sₐ+1
                        swapln(view(A, sₐ+1:tₐ-1, tₐ), view(A, sₐ, sₐ+1:tₐ-1), tₐ-sₐ-1, true)
                    end
                    swapln(view(A, sₐ, tₐ+1:n₁), view(A, tₐ, tₐ+1:n₁), n₁-tₐ)
                    αₖ[tₐ] = αₖ[icur+1]
                    αₖ[icur+1] = α₁
                    if calc_inv
                        # Mᵀ update.
                        swapln(view(M, :, sₐ), view(M, :, tₐ), n₁)
                        if i != 0
                            swapln(view(mW, sₐ, 1:δi), view(mW, tₐ, 1:δi), δi)
                        end
                    end
                end
            end
            aₖ = A[:, icur+1]
            αₖ ./= αₖ[icur+1]
            αₖ[1:icur+1] .= 0.0
            # aₖ -= vV*Wω - vW*Vω
            if i != 0
                aₖ -= (view(vV, :, 1:i) * view(vW, icur+1, 1:i) -
                       view(vW, :, 1:i) * view(vV, icur+1, 1:i))
            end
            # update M change.
            if calc_inv
                mₖ .= mW[icur+1, :]
                if i != 0
                    # change of already added terms.
                    ger!(-1.0, αₖ, view(mₖ, 1:i), view(mW, :, 1:i))
                end
                mW[:, i+1] = αₖ
            end
            vV[:, i+1] = aₖ
            vW[:, i+1] = αₖ
        end
        # Apply transformation.
        r2k!(A, vW, vV, 1.0, 1.0, k=δi)
        # Apply M-transformation.
        if calc_inv
            # TODO: change load order.
            mM[:, 1:δi] = M[:, ist+1:ist+δi]
            gemm!('N', 'T', -1.0, view(mM, :, 1:δi), view(mW, :, 1:δi), 1.0, M)
        end
        # Clear changes.
        # vV .= 0.0
        # vW .= 0.0
    end
    end # @uviews mM
    end # @uviews mW
    end # @uviews vW
    end # @uviews vV

    # Return values are the same.
    PfA = prod([A[i, i+1] for i=1:2:n₁-1])
    if calc_inv
        PfA, M*inv_tile(A)* ᵀ(M)
    else
        (PfA, )
    end
end

end
