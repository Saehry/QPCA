using LinearAlgebra: eigen, I
using FFTW
using Statistics
using MultivariateStats

#return a mod b, except when it equals 0, when we return b instead
#(useful for indexing later)
function indmod(a, b)
    m = mod(a, b)
    if m == 0
        return b
    else
        return m
    end
end

#cross-correlation of matrices A and B
function matrixXCorr(A, B)
    N, s, _ = size(A)

    Y = zeros(ComplexF64, s, s, N)
    for i = 0:(N - 1)
        Y[:, :, i + 1] = sum(map(j->@views(A[j + 1,:,:]*B[indmod(j + i + 1, N),:,:]), 
            0:(N - 1)))
    end
    return Y
end


#return the FFT of the cross-correlation of A and A^H
function matrixXCorrFFT(A)
    N, s, _ = size(A)
    AR = reverse(A; dims=1)
    ARshift = AR[indmod.(0:(N - 1), N), :, :]
    AH = permutedims(conj(A), [1, 3, 2])

    Yhat = zeros(ComplexF64, s, s, N)
    ARhat = fft(ARshift, 1)
    AHhat = fft(AH, 1)
    for i = 1:N
        Yhat[:, :, i] = ARhat[indmod(i, N), :, :]*AHhat[indmod(i, N), :, :]
    end
    return Yhat
end


#rotate elements of vector v in complex space so that the first non-zero one is purely real
function rotateVec(v)
    i = 1
    while v[i] ≈ 0
        i += 1
        if i > length(v)
            return v  #return the zero vector if it was input
        end
    end
    return conj(v[i])/abs(v[i])*v
end

rearrangePulse(pulse, N) = vcat(reshape(pulse, :, N)[:,end:-1:1]...)

#returns q and the *normalized* lambda_i's
#each column of Y is an observation y_i
#N is the number of symbol periods
#optionally provide angles ψ to address non-uniqueness
function getPulses(Y, N; ψ=nothing)
    n, m = size(Y)
    if mod(n, N) != 0
        Y = vcat(Y, zeros(N*ceil(Int, n/N) - n, m))
    end
    n = size(Y)[1]
    s = n÷N

    YSeq = @views(permutedims(reshape(Y, s, N, :), [2, 1, 3]))
    YHatSeq = matrixXCorrFFT(YSeq)
    eigs = map(j->eigen(@view(YHatSeq[:,:,j])), 1:N) #eigenvalues are in ascending order
    vecs = map(x->x.vectors, eigs)  #N elements of size s x s.  jth element is the matrix of eigenvectors of YHatSeq[:,:,j]
    vals = map(x->x.values, eigs) #N elements of length s. jth element is vector of eigenvalues of YHatSeq[:,:,j]

    maxEigs = map(i->vecs[i][:, :], 1:N)
    maxVals = map(i->vals[i][:], 1:N)
    #we know the lambdas are real, so cast to real wlog
    eigVals = map(j->real(fft(map(i->maxVals[i][end - j], 1:N))[1]), 0:(s - 1))
   
    pulses = zeros(ComplexF64, size(Y)[1], s)
    for i = 1:s
        parts = map(x->x[:,i], maxEigs)
        if isnothing(ψ)
            #force all first non-zero components to point in same direction
            rotatedMaxEigs = map(j-> rotateVec(parts[j]), 1:N)
        else
            rotatedMaxEigs = map(j->exp(im*ψ[j])*parts[j], 1:N)
        end

        out = vcat(rearrangePulse(ifft(hcat(rotatedMaxEigs...), 2), N)...)  #this has length sN
        maxI = argmax(real.(out[1:s:end]))
        pulses[:, s - i + 1] = circshift(out, -(maxI - 1)*s) 
    end
    return pulses, eigVals/sum(eigVals) #First column has first component 
end
