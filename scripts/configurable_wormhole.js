function wormhole(t, l, theta, phi)
{
    var M = 0.01;
    var p = 1;
    var a = 0.001;

    var x = 2 * (CMath.fabs(l) - a) / (Math.PI * M);

    var r = CMath.select(CMath.fabs(l) <= a,
    p,
    p + M * (x * CMath.atan(x) - 0.5 * CMath.log(1 + x * x))
    );
	
    var dt = -1;
    var dl = 1;
    var dtheta = r * r;
    var dphi = r * r * CMath.sin(theta) * CMath.sin(theta);

    return [dt, dl, dtheta, dphi];
}

wormhole