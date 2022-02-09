sktdsm!(T, B) = begin
    n, k = size(B)
    n == size(T)[1] + 1 || error("Mismatch")
    n % 2 == 0 || error("Non-invertable")

    B[1, :] ./= -T[1]
    for i = 3:2:n
        # Row i-2 -> i.
        B[i, :] .-= B[i-2, :] .* T[i-1]
        # Row i.
        B[i, :] ./= -T[i]
    end

    B[n, :] ./= T[n-1]
    for i = n-2:-2:1
        # Row i+2 -> i.
        B[i, :] .+= B[i+2, :] .* T[i]
        # Row i.
        B[i, :] ./= T[i-1]
    end

    # Swapping.
    for i = 1:2:n
        s = B[i, :]
        B[i, :] .= B[i+1, :]
        B[i+1, :] .= s
    end
end

sktdsmx(T, B) = begin
    n, k = size(B)
    n == size(T)[1] + 1 || error("Mismatch")
    n % 2 == 0 || error("Non-invertable")

    X = zeros(n, k)
    X[2, :] .= B[1, :] ./ -T[1]
    for i = 3:2:n
        # Row i-2 -> i.
        X[i+1, :] .= B[i, :] .- X[i-1, :] .* T[i-1]
        # Row i.
        X[i+1, :] ./= -T[i]
    end

    X[n-1, :] .= B[n, :] ./ T[n-1]
    for i = n-2:-2:1
        # Row i+2 -> i.
        X[i-1, :] .= B[i, :] .+ X[i+1, :] .* T[i]
        # Row i.
        X[i-1, :] ./= T[i-1]
    end

    X
end

