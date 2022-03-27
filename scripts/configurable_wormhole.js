///https://arxiv.org/pdf/1502.03809.pdf
function wormhole(t, l, theta, phi)
{
	$cfg.M.$default = 0.01;
	$cfg.p.$default = 1;
	$cfg.a.$default = 0.001;
	
    var M = $cfg.M;
    var p = $cfg.p;
    var a = $cfg.a;

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
