function func(misner_T, misner_phi, y, z)
{
	var ret = [misner_T, misner_phi, y, z];
	
	ret[1] = CMath.smooth_fmod(ret[1], $cfg.phi0);
	
	return ret;
}

func