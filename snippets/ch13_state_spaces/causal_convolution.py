def causal_convolution(u, kernel):
    """Sum the available lag terms at each input position."""
    u = _real_array(u, "u", 1)
    kernel = _real_array(kernel, "kernel", 1)
    output = np.empty(len(u))
    with np.errstate(over="raise", invalid="raise"):
        for t in range(len(u)):
            count = min(t + 1, len(kernel))
            newest_first = u[t - count + 1:t + 1][::-1]
            output[t] = kernel[:count] @ newest_first
    return output


def fft_causal_convolution(u, kernel):
    """Use a padded real FFT, then keep the first len(u) positions."""
    u = _real_array(u, "u", 1)
    kernel = _real_array(kernel, "kernel", 1)
    fft_length = len(u) + len(kernel) - 1
    with np.errstate(over="raise", invalid="raise"):
        spectrum = np.fft.rfft(u, n=fft_length) * np.fft.rfft(kernel, n=fft_length)
        full = np.fft.irfft(spectrum, n=fft_length)
    return full[:len(u)]
