function wormhole(t, p, theta, phi)
{
    var n = 1;
	
    var dt = -1;
    var dr = 1;
    var dtheta = (p * p + n * n);
    var dphi = (p * p + n * n) * (CMath.sin(theta) * CMath.sin(theta));

    return [dt, dr, dtheta, dphi];
}

wormhole