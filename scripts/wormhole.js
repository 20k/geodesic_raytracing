////https://arxiv.org/pdf/0904.4184.pdf
function wormhole(t, p, theta, phi)
{
	$cfg.n.$default = 1;
	
    var n = $cfg.n;

    var dt = -1;
    var dr = 1;
    var dtheta = (p * p + n * n);
    var dphi = (p * p + n * n) * (CMath.sin(theta) * CMath.sin(theta));

    return [dt, dr, dtheta, dphi];
}

wormhole
