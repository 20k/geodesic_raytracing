function cosmic_string(t, r, theta, phi)
{
	$cfg.rs.$default = 1;
	$cfg.B.$default = 0.3;
	
    var c = 1;

    var rs = $cfg.rs;

    var dt = -(1 - rs/r) * c * c;
    var dr = 1 / (1 - rs/r);
    var dtheta = r * r;

    var B = $cfg.B;
    var dphi = r * r * B * B * CMath.sin(theta) * CMath.sin(theta);

    return [dt, dr, dtheta, dphi];
}

cosmic_string
