function ingoing_ef_to_polar(v, r, theta, phi)
{
	var rs = $cfg.rs;
	
    var r_star = r + rs * CMath.log((r / rs) - 1);
    
	var t = v - r_star;

    return [t, r, theta, phi];
}

ingoing_ef_to_polar
