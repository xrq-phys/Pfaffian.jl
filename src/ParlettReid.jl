# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"Module for Parlett-Reid transformation."
module ParlettReid

using LinearAlgebra
using LinearAlgebra.BLAS: ger!
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
        αₖ = aₖ # A[:, i], this is subscripted k+1 in Wimmer's paper.
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

end
