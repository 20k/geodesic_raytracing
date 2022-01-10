function polar_to_cartesian(t, r, theta, phi)
{
    var x = r * CMath.sin(theta) * CMath.cos(phi);
    var y = r * CMath.sin(theta) * CMath.sin(phi);
    var z = r * CMath.cos(theta);

    return [t, x, y, z];
}

function alcubierre_distance(t, r, theta, phi)
{
    var cart = polar_to_cartesian(t, r, theta, phi);

    var dxs_t = 2;

    var x_pos = cart[1] - dxs_t * t;

    return CMath.fast_length(x_pos, cart[2], cart[3]);
}

alcubierre_distance
