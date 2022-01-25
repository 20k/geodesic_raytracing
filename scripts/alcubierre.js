///rendering alcubierre nicely is very hard: the shell is extremely thin, and flat on both sides
///this means that a naive timestepping method results in a lot of distortion
///need to crank down subambient_precision, and crank up new_max to about 20 * rs
///performance and quality is made significantly better with a dynamic timestep based on an error estimate, and then unstepping if it steps too far
function alcubierre_metric(t, x, y, z)
{
	$cfg.velocity.$default = 2;
	$cfg.sigma.$default = 1;
	$cfg.R.$default = 2;
	
    var dxs_t = $cfg.velocity;
    var xs_t = dxs_t * t;
    var vs_t = dxs_t;

    var rs_t = CMath.fast_length(x - xs_t, y, z);

    var sigma = $cfg.sigma;
    var R = $cfg.R;

    var f_rs = (CMath.tanh(sigma * (rs_t + R)) - CMath.tanh(sigma * (rs_t - R))) / (2 * CMath.tanh(sigma * R));

    var dt = (vs_t * vs_t * f_rs * f_rs - 1);
    var dxdt = -2 * vs_t * f_rs;
    var dx = 1;
    var dy = 1;
    var dz = 1;

    var ret = [];
    ret.length = 16;

    ret[0 * 4 + 0] = dt;
    ret[1 * 4 + 0] = dxdt * 0.5;
    ret[0 * 4 + 1] = dxdt * 0.5;
    ret[1 * 4 + 1] = dx;
    ret[2 * 4 + 2] = dy;
    ret[3 * 4 + 3] = dz;

    return ret;
}

alcubierre_metric
