def double_fourier_viscous_op(param, var):
    return param*var.lap*var[:]
